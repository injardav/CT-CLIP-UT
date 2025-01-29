from pathlib import Path
from shutil import rmtree
from datetime import timedelta
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from accelerate.utils import DistributedDataParallelKwargs

from utils.TrainDataset import TrainDataset
from utils.InferenceDataset import InferenceDataset
from utils.metrics import calculate_metrics, save_metrics, plot_per_class_f1, plot_precision_recall_curve, plot_roc_curve
from CTCLIP import CTCLIP
from utils.optimizer import get_optimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

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
        train_reports: str,
        valid_reports: str,
        valid_labels: str,
        train_metadata: str,
        tokenizer: BertTokenizer = None,
        lr: float = 1.25e-6,
        wd: float = 0.0,
        max_grad_norm: float = 0.5,
        save_results_every: int = 1000,
        save_model_every: int = 1000,
        results_folder: str = './results',
        num_workers: int = 8
    ):
        super().__init__()
        self.accelerator = Accelerator(
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True),
                InitProcessGroupKwargs(timeout=timedelta(seconds=36000), backend="nccl")
            ],
            mixed_precision="fp16",
            device_placement=True
        )

        log_level = logging.INFO if Accelerator().is_main_process else logging.ERROR
        logging.basicConfig(level=log_level, format="%(asctime)s - %(message)s")

        self.CTClip = CTClip
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(
            'microsoft/BiomedVLP-CXR-BERT-specialized',
            do_lower_case=True
        )

        self.steps = 0
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.train_ds = TrainDataset(data_folder=data_train, reports=train_reports, metadata=train_metadata)
        self.valid_ds = InferenceDataset(data_folder=data_valid, reports=valid_reports, labels=valid_labels)

        self.train_sampler = DistributedSampler(
            self.train_ds,
            num_replicas=self.accelerator.num_processes,  # Total GPUs
            rank=self.accelerator.process_index,  # Unique index of this GPU
            shuffle=True
        )

        self.valid_sampler = DistributedSampler(
            self.valid_ds,
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            shuffle=False  # Don't shuffle validation data
        )

        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size, sampler=self.train_sampler, num_workers=num_workers)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=1, sampler=self.valid_sampler, num_workers=num_workers)

        self.train_dl_iter = create_cyclic_dataloader(self.train_dl)
        self.valid_dl_iter = create_cyclic_dataloader(self.valid_dl)

        self.optim = get_optimizer(CTClip.parameters(), lr=lr, wd=wd)

        (
            self.train_dl_iter,
            self.valid_dl_iter,
            self.CTClip,
            self.optim,
        ) = self.accelerator.prepare(
            self.train_dl_iter,
            self.valid_dl_iter,
            self.CTClip,
            self.optim,
        )

        self.save_results_every = save_results_every
        self.save_model_every = save_model_every
        self.results_folder = Path(results_folder)
        self.best_model_path = self.results_folder / "best_model"
        self.best_score = float("inf")

        if self.accelerator.process_index == 0:
            self.results_folder.mkdir(parents=True, exist_ok=True)
            self.best_model_path.mkdir(parents=True, exist_ok=True)

            print(f"Training size: {len(self.train_ds)}")
            print(f"Validation size: {len(self.valid_ds)}")

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

        image, text = next(self.train_dl_iter)
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

        if self.accelerator.is_main_process:
            logger.info(f"Step {self.steps}: Loss = {logs['loss']}")

        if self.steps % self.save_results_every == 0:
            self.evaluate_and_save()

        if self.steps % self.save_model_every == 0 and self.accelerator.is_main_process:
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

                output, _ = self.CTClip(text_tokens, image, return_loss=False)
                output = torch.nn.functional.softmax(output, dim=0)
                predicted_class = (output[:, 0] > output[:, 1]).long()  # Compare probabilities
                predicted_labels.extend(predicted_class.tolist())

            predicted_all.append(predicted_labels)
            real_all.append(labels.cpu().numpy())

        # Gather predictions and labels from all ranks
        predicted_all = self.accelerator.gather_for_metrics(predicted_all)
        real_all = self.accelerator.gather_for_metrics(real_all)
        
        if self.accelerator.is_main_process:
            predicted_all = np.array(predicted_all)
            real_all = np.array(real_all).squeeze(axis=1)

            metrics = calculate_metrics(predicted_all, real_all, pathologies)
            save_metrics(metrics, pathologies, results_path)

            plot_precision_recall_curve(real_all, predicted_all, pathologies, results_path)
            plot_roc_curve(real_all, predicted_all, pathologies, results_path)
            plot_per_class_f1(metrics, pathologies, results_path)

            # Compute mean ROC AUC for evaluation
            macro_f1 = np.mean(metrics['macro_f1'])

            # Save best model based on ROC AUC
            if macro_f1 > self.best_score:  # Replace with `macro_f1` if preferred
                self.best_score = macro_f1
                best_model_path = self.best_model_path / f"best_checkpoint_{self.steps}.pt"
                self.save_model(best_model_path)
                logger.info(f"New best model saved at step {self.steps} with macro_f1 = {macro_f1:.4f}")

    def train(self):
        while self.steps < self.num_train_steps:
            self.train_step()
        logger.info("Training complete.")
