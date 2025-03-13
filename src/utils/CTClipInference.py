import time
import torch
import torch.nn.functional as F

from pathlib import Path
from datetime import timedelta, datetime
from scipy.ndimage import zoom

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from accelerate.utils import DistributedDataParallelKwargs

from utils.InferenceDataset import InferenceDataset
from utils.metrics import *
from utils.visualizations import Visualizations
from CTCLIP import CTCLIP


PATHOLOGIES = [
            "Medical material", "Arterial wall calcification", "Cardiomegaly",
            "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
            "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",
            "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
            "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",
            "Bronchiectasis", "Interlobular septal thickening"
        ]


class SingleSampleDistributedSampler(Sampler):
    """
    Forces all GPUs in Distributed Data Parallel (DDP) to receive the same sample.
    Useful when multiple GPU's need to be used on a computationally expensive task.
    """
    def __init__(self, dataset, sample_index=0, num_replicas=None, rank=None):
        self.sample_index = sample_index  # The single index to be used on all GPUs
        self.num_replicas = num_replicas or torch.distributed.get_world_size()
        self.rank = rank or torch.distributed.get_rank()

    def __iter__(self):
        return iter([self.sample_index])  # Return only one index for all processes

    def __len__(self):
        return 1  # Only one sample is being used   


class CTClipInference(nn.Module):
    """
    CTClipInference for evaluation of CLIP model.
    """
    def __init__(
        self,
        model: CTCLIP,
        batch_size: int,
        data_valid: str,
        valid_reports: str,
        valid_labels: str,
        valid_metadata: str,
        tokenizer: BertTokenizer = None,
        lr: float = 1e-4,
        wd: float = 0.0,
        results_folder: str = "./results",
        num_workers: int = 8,
        num_valid_samples: int = 20,
        zero_shot: bool = False,
        visualize: bool = False
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

        self.batch_size = batch_size
        self.ds = InferenceDataset(data_folder=data_valid, reports=valid_reports, metadata=valid_metadata, labels=valid_labels, num_samples=num_valid_samples)

        self.single_sampler = SingleSampleDistributedSampler(self.ds, 0)
        self.dist_sampler = DistributedSampler(
            self.ds,
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            shuffle=False,
            drop_last=True
        )

        self.single_dl = DataLoader(self.ds, batch_size=batch_size, sampler=self.single_sampler, num_workers=num_workers)
        self.dist_dl = DataLoader(self.ds, batch_size=batch_size, sampler=self.dist_sampler, num_workers=num_workers)

        (
            self.model
        ) = self.accelerator.prepare(
            self.model
        )

        self.metrics = []
        self.valid_accuracies = []
        self.best_score = 0

        self.results_folder = results_folder
        if self.accelerator.process_index == 0:
            # Create correct results directory structure with date
            self.results_folder = Path(results_folder) / datetime.now().strftime("%d-%m-%Y")
            self.results_folder.mkdir(parents=True, exist_ok=True)

        self.zero_shot = zero_shot
        self.visualize = visualize

        self.vis = Visualizations(
            self.model,
            self.accelerator,
            self.single_dl,
            self.dist_dl,
            self.batch_size,
            self.tokenizer,
            self.results_folder
        )
        
        self.maybe_print(f"Validation size: {len(self.dist_dl.dataset)}")

    def load_model(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        pkg = torch.load(path)
        self.model.load_state_dict(pkg["model"])
        self.optim.load_state_dict(pkg["optim"])
        self.model = self.model.to(self.accelerator.device)

    def validate_prompts(self, image_latents, text_latents, temp):
        """
        Compute evaluation loss based on both given text prompts.
        """
        text_latents_present = text_latents[::2]  # Shape: [B, embed_dim]
        text_latents_absent = text_latents[1::2]  # Shape: [B, embed_dim]

        # Compute separate similarity matrices for present and absent prompts.
        sim_matrix_present = image_latents @ text_latents_present.t() / temp  # Shape: [B, B]
        sim_matrix_absent  = image_latents @ text_latents_absent.t()  / temp  # Shape: [B, B]

        return sim_matrix_present, sim_matrix_absent

    def zero_shot(self):
        """
        Process a single batch and compute results.
        """
        predictions, targets = [], []

        for i, batch in enumerate(self.dist_dl):
            images, _, labels, _, _ = [b.to(self.accelerator.device) if isinstance(b, torch.Tensor) else b for b in batch]
            predicted_labels = torch.zeros(len(PATHOLOGIES), dtype=torch.long, device=self.accelerator.device)

            for j, pathology in enumerate(PATHOLOGIES):
                text_tokens = self.tokenizer(
                    [f"There is {pathology}.", f"There is no {pathology}."],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                ).to(self.accelerator.device)

                # Forward pass of text prompts for pathology predictions
                self.model.zero_grad()
                with torch.enable_grad():
                    sim_matrix, image_latents, text_latents, temp, _, _ = self.model(text_tokens, images)
                sim_matrix_present, sim_matrix_absent = self.validate_prompts(image_latents, text_latents, temp)

                # Extract diagonal elements to get predictions for each sample
                present_scores = torch.diag(sim_matrix_present)  # Present prompt scores
                absent_scores = torch.diag(sim_matrix_absent)    # Absent prompt scores

                # Apply softmax to the extracted diagonal scores, compute and save predictions
                probabilities = torch.nn.functional.softmax(torch.stack([present_scores, absent_scores], dim=1), dim=1)
                present_scores, absent_scores = probabilities[:, 0], probabilities[:, 1]
                predicted_class = (present_scores > absent_scores).long()
                device_prediction = predicted_class[self.accelerator.process_index]
                predicted_labels[j] = device_prediction.item()

            predictions.append(predicted_labels)
            targets.append(labels)

        # Convert predictions and targets to tensors
        predictions = torch.stack(predictions)
        targets = torch.stack(targets)

        # Gather across devices
        predictions, targets = self.accelerator.gather_for_metrics((predictions, targets))
        targets = targets.squeeze(dim=1).cpu().numpy()
        predictions = predictions.cpu().numpy()

        # Compute evaluation metrics
        if self.accelerator.is_main_process:
            metrics = calculate_metrics(predictions, targets, PATHOLOGIES)
            self.valid_accuracies.append(metrics["mAP"])
            self.metrics.append(metrics)
            save_metrics(self.metrics, PATHOLOGIES, self.results_folder)

            # Plot metrics
            plot_precision_recall_curve(targets, predictions, PATHOLOGIES, self.results_folder)
            plot_roc_curve(targets, predictions, PATHOLOGIES, self.results_folder)
            plot_per_class_f1(metrics, PATHOLOGIES, self.results_folder)
            plot_all_metrics(self.metrics, self.results_folder)

    def infer(self):
        """
        Runs distributed inference.
        """
        start_time = time.time()
        self.maybe_print("Evaluation started")
    
        self.model.eval()

        if self.zero_shot:
            self.zero_shot()

        if self.visualize:
            self.vis.visualize(
                attention=True,
                grad_cam=True,
                occlusion=True
            )
            
        total_time = time.time() - start_time
        self.maybe_print(f"Evaluation completed. Total Evaluation Time: {str(timedelta(seconds=total_time))}")
