import os
import torch
import numpy as np
from pathlib import Path
from einops import rearrange
from transformers import BertTokenizer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformer_maskgit.optimizer import get_optimizer
from data_inference import CTReportDatasetinfer
from eval import evaluate_internal
import pandas as pd
import tqdm


class CTClipInference(torch.nn.Module):
    def __init__(
        self,
        CTClip,
        *,
        num_train_steps,
        batch_size,
        data_folder,
        reports_file,
        lr=1e-4,
        wd=0.0,
        max_grad_norm=0.5,
        save_results_every=100,
        save_model_every=2000,
        results_folder="./results",
        labels="labels.csv",
        save_weights=False,
        accelerate_kwargs=None
    ):
        super().__init__()
        accelerate_kwargs = accelerate_kwargs or {}
        self.accelerator = Accelerator(
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
            **accelerate_kwargs,
        )

        # Core attributes
        self.CTClip = CTClip
        self.tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True)
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.save_weights = save_weights
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.register_buffer("steps", torch.tensor(0.0))

        # Dataset and optimizer
        self.dataset = CTReportDatasetinfer(data_folder=data_folder, csv_file=reports_file, labels=labels)
        self.dataloader = self.prepare_dataloader(self.dataset)
        self.optim = get_optimizer(CTClip.parameters(), lr=lr, wd=wd)
        self.lr_scheduler = self.configure_lr_scheduler(self.optim, lr)

        # Prepare model and training components
        self.device = self.accelerator.device
        self.CTClip.to(self.device)
        self.prepare_training_components()

    def prepare_dataloader(self, dataset):
        """Prepare the DataLoader."""
        return DataLoader(
            dataset,
            num_workers=6,
            batch_size=1,
            shuffle=True,
        )

    def configure_lr_scheduler(self, optimizer, lr):
        """Configure the learning rate scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=4000000,  # Maximum iterations
            T_mult=1,
            eta_min=lr / 100,
        )

    def prepare_training_components(self):
        """Prepare components with the Accelerator."""
        self.dataloader, self.CTClip, self.optim, self.lr_scheduler = self.accelerator.prepare(
            self.dataloader, self.CTClip, self.optim, self.lr_scheduler
        )

    def save_model(self, path):
        """Save the model and optimizer state."""
        if not self.accelerator.is_local_main_process:
            return

        torch.save(
            {
                "model": self.accelerator.get_state_dict(self.CTClip),
                "optimizer": self.optim.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        """Load the model and optimizer state."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        self.CTClip.load_state_dict(checkpoint["model"])
        self.optim.load_state_dict(checkpoint["optimizer"])

    def log_attention_weights(self, output, weights_dir, pathology_name, i):
        """Save attention weights and feature maps."""
        Path(weights_dir).mkdir(parents=True, exist_ok=True)
        np.savez(
            f"{weights_dir}/attention_weights_{i}.npz",
            text_attention_weights=output["text_attention_weights"],
            spatial_attention_weights=output["spatial_attention_weights"],
            temporal_attention_weights=output["temporal_attention_weights"],
        )

    def compute_gradcam(self, gradients, features):
        """Compute Grad-CAM weights and maps."""
        gradients_pooled = gradients.mean(dim=-1)
        gradients_pooled = gradients_pooled / (gradients_pooled.norm(dim=1, keepdim=True) + 1e-8)
        weighed_features = gradients_pooled.unsqueeze(-1) * features
        grad_cam_map = F.relu(weighed_features.sum(dim=2))  # ReLU and sum over embedding dimension
        return grad_cam_map

    def train_step(self):
        """Perform a single training step."""
        logs = {}
        self.CTClip.eval()  # Set model to eval mode for inference
        predictions, real_labels = [], []

        # Iterate over the dataset
        for i, (image, text, labels) in enumerate(tqdm.tqdm(self.dataloader)):
            image, text = image.to(self.device), text.to(self.device)

            with torch.no_grad():
                output, attention_weights = self.CTClip(text, image, return_attention=True)

            predictions.append(output.cpu().numpy())
            real_labels.append(labels.cpu().numpy())

        # Save predictions and metrics
        np.savez(f"{self.results_folder}/predictions.npz", data=predictions)
        np.savez(f"{self.results_folder}/real_labels.npz", data=real_labels)

        logs["predictions"] = predictions
        logs["real_labels"] = real_labels
        return logs

    def infer(self, log_fn=noop):
        """Run inference until the maximum number of steps is reached."""
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)
            self.steps += 1

        self.accelerator.print("Inference complete.")