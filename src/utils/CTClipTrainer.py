from pathlib import Path
from shutil import rmtree
from datetime import timedelta
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from accelerate.state import DistributedDataParallelKwargs

from src.utils.TrainDataset import TrainDataset
from src.utils.InferenceDataset import InferenceDataset
from src.utils.metrics import calculate_metrics, save_metrics, plot_per_class_f1, plot_precision_recall_curve, plot_roc_curve
from src import CTCLIP
from src.utils.optimizer import get_optimizer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def apply_softmax(array):
    """
    Applies softmax function to a tensor.

    Args:
        array (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor after applying softmax.
    """
    return torch.nn.functional.softmax(array, dim=0)

def tensor_to_nifti(tensor, path, affine=np.eye(4)):
    """
    Save a tensor as a NIfTI file.

    Args:
        tensor (torch.Tensor): Input tensor with shape (D, H, W) or (C, D, H, W).
        path (str): Path to save the NIfTI file.
        affine (np.ndarray): Affine matrix for the NIfTI file.
    """
    from nibabel import Nifti1Image, save

    tensor = tensor.cpu()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Save the first channel if multi-channel
    numpy_data = tensor.permute(2, 1, 0).detach().numpy().astype(np.float32)  # Adjust for NIfTI axes
    nifti_img = Nifti1Image(numpy_data, affine)
    save(nifti_img, path)

def create_cyclic_dataloader(dataloader):
    """Creates a cyclic iterator for a DataLoader."""
    while True:
        yield from dataloader

class CTClipTrainer(nn.Module):
    """
    Trainer for the CTCLIP model, leveraging Accelerate for distributed training.
    """

    def __init__(
        self,
        CTClip: CTCLIP,
        num_train_steps: int,
        batch_size: int,
        data_train: str,
        data_valid: str,
        reports_file_train: str,
        reports_file_valid: str,
        labels: str,
        train_metadata: str,
        tokenizer: BertTokenizer = None,
        lr: float = 1.25e-6,
        wd: float = 0.0,
        max_grad_norm: float = 0.5,
        save_results_every: int = 1000,
        save_model_every: int = 1000,
        results_folder: str = './results',
        num_workers: int = 8,
        accelerate_kwargs: dict = None
    ):
        super().__init__()
        self.accelerator = Accelerator(
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True),
                InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
            ],
            **(accelerate_kwargs or {})
        )

        self.CTClip = CTClip
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(
            'microsoft/BiomedVLP-CXR-BERT-specialized',
            do_lower_case=True
        )

        self.steps = 0
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.ds = TrainDataset(data_folder=data_train, csv_file=reports_file_train, train_metadata=train_metadata)
        self.valid_ds = InferenceDataset(data_folder=data_valid, csv_file=reports_file_valid, labels=labels)

        self.dl = DataLoader(self.ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=1, shuffle=False, num_workers=num_workers)

        self.dl_iter = create_cyclic_dataloader(self.dl)
        self.valid_dl_iter = create_cyclic_dataloader(self.valid_dl)

        self.optim = get_optimizer(CTClip.parameters(), lr=lr, wd=wd)

        (
            self.dl_iter,
            self.valid_dl_iter,
            self.CTClip,
            self.optim,
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.valid_dl_iter,
            self.CTClip,
            self.optim,
        )

        self.save_results_every = save_results_every
        self.save_model_every = save_model_every
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        rmtree(str(self.results_folder))

    def save_model(self, path):
        if self.accelerator.is_local_main_process:
            pkg = {
                'model': self.accelerator.get_state_dict(self.CTClip),
                'optim': self.optim.state_dict()
            }
            torch.save(pkg, path)

    def load_model(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        pkg = torch.load(path)
        self.CTClip.load_state_dict(pkg['model'])
        self.optim.load_state_dict(pkg['optim'])

    def train_step(self):
        self.CTClip.train()
        logs = {}

        image, text = next(self.dl_iter)
        image = image.to(self.accelerator.device)
        text_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).to(self.accelerator.device)

        with self.accelerator.autocast():
            loss = self.CTClip(text_tokens, image, return_loss=True)

        self.accelerator.backward(loss)
        if self.max_grad_norm:
            self.accelerator.clip_grad_norm_(self.CTClip.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        logs['loss'] = loss.item()
        logger.info(f"Step {self.steps}: Loss = {logs['loss']}")

        if self.steps % self.save_results_every == 0:
            self.evaluate_and_save()

        if self.steps % self.save_model_every == 0:
            model_path = self.results_folder / f'CTClip.{self.steps}.pt'
            self.save_model(model_path)
            logger.info(f"Model saved at {model_path}")

        self.steps += 1
        return logs

    def evaluate_and_save(self):
        """Evaluate the model on validation data and save results."""
        self.CTClip.eval()
        pathologies = [
            'Medical material', 'Arterial wall calcification', 'Cardiomegaly',
            'Pericardial effusion', 'Coronary artery wall calcification', 'Hiatal hernia',
            'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule',
            'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
            'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation',
            'Bronchiectasis', 'Interlobular septal thickening'
        ]

        results_path = self.results_folder / f"CTClip_{self.steps}"
        results_path.mkdir(parents=True, exist_ok=True)

        predicted_all, real_all = [], []

        for _ in range(10):
            image, _, labels, _ = next(self.valid_dl_iter)
            image = image.to(self.accelerator.device)
            predicted_labels = []

            for pathology in pathologies:
                text = [f"There is {pathology}.", f"There is no {pathology}."]
                text_tokens = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                ).to(self.accelerator.device)

                output = self.CTClip(text_tokens, image, return_loss=False)
                output = apply_softmax(output)
                predicted_labels.append(float(output[0].item() > output[1].item()))

            predicted_all.append(predicted_labels)
            real_all.append(labels.cpu().numpy())

        predicted_all = np.array(predicted_all)
        real_all = np.array(real_all)

        metrics = calculate_metrics(predicted_all, real_all, pathologies)
        save_metrics(metrics, pathologies, results_path)

        plot_precision_recall_curve(real_all, predicted_all, pathologies, results_path)
        plot_roc_curve(real_all, predicted_all, pathologies, results_path)
        plot_per_class_f1(metrics, pathologies, results_path)

    def train(self):
        while self.steps < self.num_train_steps:
            self.train_step()
        logger.info("Training complete.")
