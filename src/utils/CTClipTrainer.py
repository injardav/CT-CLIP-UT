import time
import torch
import torch.nn.functional as F

from pathlib import Path
from datetime import timedelta, datetime

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from accelerate.utils import DistributedDataParallelKwargs

from utils.TrainDataset import TrainDataset
from utils.InferenceDataset import InferenceDataset
from utils.metrics import *
from models.ctclip import CTCLIP
from utils.optimizer import get_optimizer


PATHOLOGIES = [
            "Medical material", "Arterial wall calcification", "Cardiomegaly",
            "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
            "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",
            "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
            "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",
            "Bronchiectasis", "Interlobular septal thickening"
        ]


class CTClipTrainer(nn.Module):
    """
    Trainer for the CTCLIP model, leveraging Accelerate for distributed training.
    """

    def __init__(
        self,
        model: CTCLIP,
        batch_size: int,
        data_train: str,
        data_valid: str,
        train_reports: str,
        valid_reports: str,
        valid_labels: str,
        train_metadata: str,
        valid_metadata: str,
        tokenizer: BertTokenizer = None,
        lr: float = 1.25e-5,
        wd: float = 0.0,
        max_grad_norm: float = 0.5,
        results_folder: str = "./results",
        num_workers: int = 8,
        num_epochs: int = 10,
        num_save_split: int = 5,
        num_train_samples: int = 100,
        num_valid_samples: int = 20,
        save_best_model: bool = False
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
        self.maybe_print = print if self.accelerator.is_main_process else lambda *args, **kwargs: None

        self.model = model
        self.model.accelerator = self.accelerator
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            do_lower_case=True
        )

        self.num_epochs = num_epochs
        self.num_save_split = num_save_split
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.save_best_model = save_best_model

        self.train_ds = TrainDataset(data_folder=data_train, reports=train_reports, metadata=train_metadata, num_samples=num_train_samples)
        self.valid_ds = InferenceDataset(data_folder=data_valid, reports=valid_reports, metadata=valid_metadata, labels=valid_labels, num_samples=num_valid_samples)

        self.train_sampler = DistributedSampler(
            self.train_ds,
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            shuffle=True,
            drop_last=True
        )

        self.valid_sampler = DistributedSampler(
            self.valid_ds,
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            shuffle=False,
            drop_last=True
        )

        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size, sampler=self.train_sampler, num_workers=num_workers)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=batch_size, sampler=self.valid_sampler, num_workers=num_workers)

        self.optim = get_optimizer(model.parameters(), lr=lr, wd=wd)

        (
            self.model,
            self.optim,
        ) = self.accelerator.prepare(
            self.model,
            self.optim,
        )

        self.metrics = []
        self.train_losses = {}
        self.valid_losses = []
        self.best_score = float('inf')

        if self.accelerator.process_index == 0:
            # Create correct results directory structure dynamically
            self.results_folder = Path(results_folder) / datetime.now().strftime("%d-%m-%Y")
            self.results_folder.mkdir(parents=True, exist_ok=True)

            existing_folders = [d for d in self.results_folder.iterdir() if d.is_dir()]
            idx = len(existing_folders) + 1

            self.results_folder = self.results_folder / str(idx)
            self.results_folder.mkdir(parents=True, exist_ok=True)

            self.maybe_print(f"Training size: {len(self.train_dl.dataset)}")
            self.maybe_print(f"Validation size: {len(self.valid_dl.dataset)}")

    def save_model(self, name, log=None):
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            pkg = {
                "model": self.accelerator.get_state_dict(unwrapped_model),
                "optim": self.optim.state_dict()
            }
            torch.save(pkg, str(self.results_folder / name))
            with open(self.results_folder / "architecture.txt", "w") as f:
                f.write(str(unwrapped_model))

    def load_model(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        pkg = torch.load(path)
        self.model.load_state_dict(pkg["model"])
        self.optim.load_state_dict(pkg["optim"])
        self.model = self.model.to(self.accelerator.device)

    def avg_device_loss(self, loss):
        """
        Gathers loss accross devices and returns the average.
        """
        loss_tensor = torch.tensor(loss, device=self.accelerator.device)
        gathered_losses = self.accelerator.gather_for_metrics(loss_tensor)
        return gathered_losses.mean().item()

    def loss_function(self, sim_matrix, targets=None):
        """
        Compute contrastive loss using similarity matrix.
        """
        if targets is None:
            targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        loss_i2t = F.cross_entropy(sim_matrix, targets)
        loss_t2i = F.cross_entropy(sim_matrix.t(), targets)
        loss = (loss_i2t + loss_t2i) / 2

        return loss

    def train_step(self, batch):
        """
        Performs a single training step.
        """
        self.model.train()
        self.optim.zero_grad()
        
        images, texts = batch
        images = images.to(self.accelerator.device)
        text_tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).to(self.accelerator.device)

        sim_matrix, *_ = self.model(text_tokens, images)
        loss = self.loss_function(sim_matrix)
        self.accelerator.backward(loss)

        # Clip gradients if necessary
        if self.max_grad_norm and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optim.step()

        return loss.item()

    def evaluate(self, epoch):
        """
        Evaluates the model, computes combined loss, and plots results.
        """
        self.model.eval()
        total_val_loss = 0.0

        self.maybe_print(f"Evaluating epoch {epoch}")

        for i, batch in enumerate(self.valid_dl):
            images, texts, labels, _ = [b.to(self.accelerator.device) if isinstance(b, torch.Tensor) else b for b in batch]
            predicted_labels = torch.zeros(len(PATHOLOGIES), dtype=torch.long, device=self.accelerator.device)

            text_tokens = self.tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            ).to(self.accelerator.device)

            # Forward pass with original text for true validation loss
            sim_matrix, *_ = self.model(text_tokens, images)
            val_loss = self.loss_function(sim_matrix)
            total_val_loss += val_loss.item()

        # Compute average validation loss
        avg_val_loss = self.avg_device_loss(total_val_loss / len(self.valid_dl))
        self.valid_losses.append(avg_val_loss)
        self.maybe_print(f"Epoch {epoch} - Validation Loss: {avg_val_loss:.4f}")

        # Check for best score and plot losses
        if self.accelerator.is_main_process:
            if epoch == 0 or (avg_val_loss < self.best_score and self.save_best_model):
                self.best_score = avg_val_loss
                self.save_model(
                    "best_checkpoint.pt",
                    f"New best model saved at epoch {epoch}, with average validation loss {avg_val_loss:.4f}"
                )

            plot_training_progress(
                self.train_losses,
                self.valid_losses,
                self.results_folder
            )

    def train(self):
        """
        Runs distributed training.
        """
        start_time = time.time()
        save_at = len(self.train_dl) // self.num_save_split
        self.maybe_print("Training started")

        epoch_durations = []

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            self.maybe_print(f"\nStarting Epoch {epoch}/{self.num_epochs}")
            self.train_sampler.set_epoch(epoch)
            total_loss = 0.0
        
            for step, batch in enumerate(self.train_dl, start=1):
                with self.accelerator.autocast():
                    loss = self.train_step(batch)
                total_loss += loss
                avg_step_loss = self.avg_device_loss(loss)

                if step % save_at == 0:
                    self.train_losses.setdefault("steps", []).append(avg_step_loss)
                    
                # First epoch"s first step loss is also first epoch 0 loss
                if epoch == 1 and step == 1:
                    self.train_losses.setdefault("epochs", []).append(avg_step_loss)
                    self.train_losses.setdefault("steps", []).append(avg_step_loss)
                    self.evaluate(0)

                self.maybe_print(f"Epoch {epoch} | Step {step}/{len(self.train_dl)} | Avg Loss: {avg_step_loss:.6f}")

            avg_epoch_loss = self.avg_device_loss(total_loss / len(self.train_dl))
            self.train_losses.setdefault("epochs", []).append(avg_epoch_loss)

            epoch_time = time.time() - epoch_start
            epoch_durations.append(epoch_time)

            self.maybe_print(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.6f}")
            self.maybe_print(f"Time taken for Epoch {epoch}: {str(timedelta(seconds=epoch_time))}")
            
            eval_start = time.time()
            self.evaluate(epoch)
            eval_time = time.time() - eval_start
            self.maybe_print(f"Time taken for evaluation: {str(timedelta(seconds=eval_time))}")

        total_time = time.time() - start_time
        avg_epoch_time = sum(epoch_durations) / len(epoch_durations)

        self.maybe_print("Training completed")
        self.maybe_print(f"Total Training Time: {str(timedelta(seconds=total_time))}")
        self.maybe_print(f"Average Epoch Duration: {str(timedelta(seconds=avg_epoch_time))}")
