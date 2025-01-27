import os
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import BertTokenizer
from src.utils import InferenceDataset
from src.utils.optimizer import get_optimizer
from src.utils.metrics import (
    calculate_metrics, save_metrics, plot_per_class_f1,
    plot_precision_recall_curve, plot_roc_curve
)
import numpy as np
import tqdm
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler

# Helper Functions
def log_intermediary_values(content, log_file_path="/users/injarabi/output.log"):
    """Log intermediary values to a specified file."""
    with open(os.path.abspath(log_file_path), "a") as log_file:
        print(content, file=log_file)

def cycle(data_loader):
    """Cycle through the data loader indefinitely."""
    while True:
        for data in data_loader:
            yield data

def apply_softmax(tensor):
    """Apply softmax to a tensor along the first dimension."""
    return torch.nn.functional.softmax(tensor, dim=0)

class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    """Custom learning rate scheduler with warm-up and restarts."""
    def __init__(self, optimizer, initial_period, period_multiplier=1, max_lr=0.1, warmup_steps=10000, decay_factor=1.0, last_epoch=-1):
        self.initial_period = initial_period
        self.period_multiplier = period_multiplier
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.min_lr = 0
        self.iteration = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.iteration < self.warmup_steps:
            lr = self.max_lr * self.iteration / self.warmup_steps
        else:
            current_period = self.iteration - self.warmup_steps
            period_length = self.initial_period
            while current_period >= period_length:
                current_period -= period_length
                period_length *= self.period_multiplier
                self.min_lr = self.max_lr * (self.decay_factor ** (self.iteration // period_length))
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * current_period / period_length))
        self.iteration += 1
        return [lr for _ in self.optimizer.param_groups]

class CTClipInference(nn.Module):
    """CTClipInference for training and evaluation of CT models."""
    def __init__(
        self, ct_clip, *, num_train_steps, batch_size, data_folder, reports_file,
        learning_rate=1e-4, weight_decay=0., max_grad_norm=0.5, save_results_every=100,
        save_model_every=2000, results_folder='./results', labels_file="labels.csv",
        accelerate_kwargs=None
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
        self.register_buffer('steps', torch.Tensor([0]))

        # Training parameters
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate

        # Optimizer and dataset
        all_parameters = set(ct_clip.parameters())
        self.optimizer = get_optimizer(all_parameters, lr=learning_rate, wd=weight_decay)
        self.dataset = InferenceDataset(data_folder=data_folder, csv_file=reports_file, labels=labels_file)
        self.data_loader = DataLoader(self.dataset, num_workers=6, batch_size=1, shuffle=True)
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
        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

    def save(self, path):
        """Save model and optimizer state."""
        if not self.accelerator.is_local_main_process:
            return
        torch.save(
            {
                'model': self.accelerator.get_state_dict(self.ct_clip),
                'optimizer': self.optimizer.state_dict()
            }, path
        )

    def load(self, path):
        """Load model and optimizer state from a file."""
        path = Path(path)
        assert path.exists(), "Checkpoint not found."
        package = torch.load(path)
        self.accelerator.unwrap_model(self.ct_clip).load_state_dict(package['model'])
        self.optimizer.load_state_dict(package['optimizer'])

    def train_step(self):
        """Perform one training step."""
        device = self.device
        steps = int(self.steps.item())

        results_path = self.results_folder / f"CTClip_{self.steps}"
        results_path.mkdir(parents=True, exist_ok=True)

        pathologies = [
            'Medical material', 'Arterial wall calcification', 'Cardiomegaly',
            'Pericardial effusion', 'Coronary artery wall calcification', 'Hiatal hernia',
            'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule', 'Lung opacity',
            'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern',
            'Peribronchial thickening', 'Consolidation', 'Bronchiectasis', 'Interlobular septal thickening'
        ]

        self.ct_clip.eval()
        predicted_all, real_all, accession_names = [], [], []

        for _ in tqdm.tqdm(range(1)):
            valid_data, text, onehot_labels, acc_name = next(self.data_loader_iter)

            sim_score_attention = []
            for pathology in pathologies:
                log_intermediary_values(f"Evaluating pathology: {pathology}")
                text_tokens = self.tokenizer(
                    [f"{pathology} is present.", f"{pathology} is not present."],
                    return_tensors="pt", padding="max_length", truncation=True, max_length=512
                ).to(device)

                with torch.no_grad():
                    output, spatial_scores = self.ct_clip(text_tokens, valid_data.cuda(), device=device)
                    sim_score_attention.append(spatial_scores.cpu().numpy())

                output = apply_softmax(output)
                predicted_all.append(output.cpu().numpy()[0])
                real_all.append(onehot_labels.cpu().numpy()[0])
                accession_names.append(acc_name[0])

        predicted_all = np.array(predicted_all)
        real_all = np.array(real_all)

        # Calculate and save metrics
        metrics = calculate_metrics(predicted_all, real_all, pathologies)
        save_metrics(metrics, pathologies, results_path)

        # Generate and save plots
        plot_precision_recall_curve(real_all, predicted_all, pathologies, results_path)
        plot_roc_curve(real_all, predicted_all, pathologies, results_path)
        plot_per_class_f1(metrics, pathologies, results_path)

        self.steps += 1
        return {}

    def infer(self, log_fn=lambda x: None):
        """Run inference until the specified number of steps."""
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)
        self.print("Inference complete")

    def print(self, message):
        """Print a message using the accelerator."""
        self.accelerator.print(message)

    @property
    def is_main(self):
        """Check if the current process is the main process."""
        return self.accelerator.is_main_process
