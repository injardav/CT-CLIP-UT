import os
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import BertTokenizer
from utils.InferenceDataset import InferenceDataset
from utils.optimizer import get_optimizer
from utils.metrics import (
    calculate_metrics, save_metrics, plot_per_class_f1,
    plot_precision_recall_curve, plot_roc_curve
)
import numpy as np
from typing import List, Dict
import torch.optim.lr_scheduler as lr_scheduler


# Helper Functions
def log_intermediary_values(content: str, log_file_path: str = "/users/injarabi/output.log") -> None:
    """Log intermediary values to a specified file."""
    with open(os.path.abspath(log_file_path), "a") as log_file:
        print(content, file=log_file)


def cycle(data_loader: DataLoader):
    """Cycle through the data loader indefinitely."""
    while True:
        for data in data_loader:
            yield data


class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    """Custom learning rate scheduler with warm-up and restarts."""
    def __init__(self, optimizer, initial_period, period_multiplier=1, max_lr=0.1, warmup_steps=10000, decay_factor=1.0, last_epoch=-1):
        self.initial_period = initial_period
        self.period_multiplier = period_multiplier
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        iteration = self.last_epoch
        if iteration < self.warmup_steps:
            lr = self.max_lr * iteration / self.warmup_steps
        else:
            period_length = self.initial_period
            period_start = self.warmup_steps
            while iteration >= period_start + period_length:
                period_start += period_length
                period_length *= self.period_multiplier
            progress = (iteration - period_start) / period_length
            lr = self.max_lr * (self.decay_factor ** (iteration // period_length))
            lr = lr * (0.5 * (1 + math.cos(math.pi * progress)))
        return [lr for _ in self.optimizer.param_groups]


class CTClipInference(nn.Module):
    """CTClipInference for training and evaluation of CT models."""
    def __init__(
        self,
        ct_clip: nn.Module,
        num_train_steps: int,
        batch_size: int,
        data_folder: str,
        reports_file: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.,
        max_grad_norm: float = 0.5,
        save_results_every: int = 100,
        results_folder: str = './results',
        labels_file: str = "labels.csv",
        accelerate_kwargs: Dict = None
    ):
        super().__init__()
        accelerate_kwargs = accelerate_kwargs or {}
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.ct_clip = ct_clip
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)

        # Initialize paths and results directory
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # Parameters
        self.steps = 0
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate

        # Optimizer and dataset
        self.optimizer = get_optimizer(set(ct_clip.parameters()), lr=learning_rate, wd=weight_decay)
        self.dataset = InferenceDataset(data_folder, reports_file, labels_file)
        self.data_loader = DataLoader(self.dataset, num_workers=6, batch_size=batch_size, shuffle=True)
        self.data_loader_iter = cycle(self.data_loader)

        # Device and scheduler
        self.device = self.accelerator.device
        self.ct_clip.to(self.device)
        self.lr_scheduler = CosineAnnealingWarmUpRestarts(
            self.optimizer, initial_period=4000000, warmup_steps=10000, max_lr=learning_rate
        )

        # Accelerator preparation
        (
            self.data_loader_iter,
            self.ct_clip,
            self.optimizer,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.data_loader_iter,
            self.ct_clip,
            self.optimizer,
            self.lr_scheduler
        )

        # Save intervals
        self.save_results_every = save_results_every

    def save(self, path: str) -> None:
        """Save model and optimizer state."""
        if not self.accelerator.is_local_main_process:
            return
        torch.save(
            {
                'model': self.accelerator.get_state_dict(self.ct_clip),
                'optimizer': self.optimizer.state_dict()
            }, Path(path)
        )

    def load(self, path: str) -> None:
        """Load model and optimizer state from a file."""
        path = Path(path)
        assert path.exists(), "Checkpoint not found."
        package = torch.load(path)
        self.accelerator.unwrap_model(self.ct_clip).load_state_dict(package['model'])
        self.optimizer.load_state_dict(package['optimizer'])

    def _process_batch(self, pathologies: List[str]):
        """Process a single batch and log results."""
        predicted_all, real_all = [], []
        image, _, labels, acc_name = next(self.data_loader_iter)
        for pathology in pathologies:
            log_intermediary_values(f"Processing {pathology}")
            text_tokens = self.tokenizer(
                [f"{pathology} is present.", f"{pathology} is not present."],
                return_tensors="pt", padding="max_length", truncation=True, max_length=512
            ).to(self.device)

            with torch.no_grad():
                output, _ = self.ct_clip(text_tokens, image.cuda())
                output = torch.nn.functional.softmax(output, dim=0)

            predicted_all.append(output.cpu().numpy()[0])
            real_all.append(labels.cpu().numpy()[0])

        return np.array(predicted_all), np.array(real_all)

    def infer_step(self) -> None:
        """Perform one inference step."""
        pathologies = [
            'Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',
            'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy', 'Emphysema',
            'Atelectasis', 'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
            'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
            'Interlobular septal thickening'
        ]
        predicted_all, real_all = self._process_batch(pathologies)

        # Calculate metrics and save results
        metrics = calculate_metrics(predicted_all, real_all, pathologies)
        save_metrics(metrics, pathologies, self.results_folder)
        plot_precision_recall_curve(real_all, predicted_all, pathologies, self.results_folder)
        plot_roc_curve(real_all, predicted_all, pathologies, self.results_folder)
        plot_per_class_f1(metrics, pathologies, self.results_folder)

        self.steps += 1

    def infer(self) -> None:
        """Run inference until the specified number of steps."""
        while self.steps < self.num_train_steps:
            self.infer_step()
        self.print("Inference complete")

    def print(self, message: str) -> None:
        """Print a message using the accelerator."""
        self.accelerator.print(message)
