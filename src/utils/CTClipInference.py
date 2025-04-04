import time
import torch
import torch.nn.functional as F

from pathlib import Path
from datetime import timedelta, datetime
from scipy.ndimage import zoom

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from accelerate.utils import DistributedDataParallelKwargs

from utils.InferenceDataset import InferenceDataset
from utils.metrics import *
from utils.visualizations import Visualizations
from models.ctclip import CTCLIP


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
        if torch.distributed.is_initialized():
            self.num_replicas = num_replicas or torch.distributed.get_world_size()
            self.rank = rank or torch.distributed.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

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
        results_folder: str = "./results",
        num_workers: int = 8,
        num_valid_samples: int = 0,
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
        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        self.maybe_print = print if self.accelerator.is_main_process else lambda *args, **kwargs: None

        self.model = model
        self.model.accelerator = self.accelerator
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            do_lower_case=True
        )

        self.num_valid_samples = num_valid_samples if num_valid_samples else (1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())
        self.batch_size = batch_size
        self.ds = InferenceDataset(data_folder=data_valid, reports=valid_reports, metadata=valid_metadata, labels=valid_labels, num_samples=self.num_valid_samples)
        
        self.single_dist_sampler = SingleSampleDistributedSampler(
            self.ds, 
            sample_index=0,
            num_replicas=self.world_size,
            rank=self.rank
        )

        if self.world_size > 1:
            self.sampler = DistributedSampler(
                self.ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=True
            )
        else:
            self.sampler = RandomSampler(self.ds)

        self.single_dist_dl = DataLoader(self.ds, batch_size=batch_size, sampler=self.single_dist_sampler, num_workers=num_workers)
        self.dl = DataLoader(self.ds, batch_size=batch_size, sampler=self.sampler, num_workers=num_workers)

        (
            self.model
        ) = self.accelerator.prepare(
            self.model
        )

        self.model.eval()

        self.metrics = []
        self.results_folder = results_folder
        if self.accelerator.is_main_process:
            # Create correct results directory structure with date
            self.results_folder = Path(results_folder) / datetime.now().strftime("%d-%m-%Y")
            self.results_folder.mkdir(parents=True, exist_ok=True)

        self.zero_shot = zero_shot
        self.visualize = visualize

        self.vis = Visualizations(
            self.model,
            self.accelerator,
            self.single_dist_dl,
            self.dl,
            self.batch_size,
            self.results_folder,
            self.tokenizer
        )
        
        self.maybe_print(f"Validation size: {len(self.dl.dataset)}")

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
        sim_matrix_present = image_latents @ text_latents_present.t() * temp  # Shape: [B, B]
        sim_matrix_absent  = image_latents @ text_latents_absent.t()  * temp  # Shape: [B, B]

        return sim_matrix_present, sim_matrix_absent

    def zeroshot(self):
        """
        Process a single batch and compute results.
        """
        softmax = nn.Softmax(dim=0)
        pos_softmax_predictions, targets = [], []

        for i, batch in enumerate(self.dl):
            images, _, labels, _, _ = [b.to(self.accelerator.device) if isinstance(b, torch.Tensor) else b for b in batch]
            labels = labels[0] # unwrap labels from batch list
            predicted_labels = torch.zeros(len(PATHOLOGIES), dtype=torch.double, device=self.accelerator.device)

            for j, pathology in enumerate(PATHOLOGIES):
                text_tokens = self.tokenizer(
                    [f"There is {pathology}.", f"There is no {pathology}."],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                ).to(self.accelerator.device)

                # Forward pass of text prompts for pathology predictions
                with torch.no_grad():
                    _, image_latents, text_latents, temp, _, _ = self.model(text_tokens, images)
                sim_matrix_present, sim_matrix_absent = self.validate_prompts(image_latents, text_latents, temp)

                # Extract diagonal elements to get device predictions for each sample
                present_score = torch.diag(sim_matrix_present)[self.accelerator.process_index]  # Present prompt score for device
                absent_score = torch.diag(sim_matrix_absent)[self.accelerator.process_index]    # Absent prompt score for device

                # Compute and save positive prediction probability
                softmax_probs = softmax(torch.stack([present_score, absent_score]))
                predicted_labels[j] = softmax_probs[0]

            pos_softmax_predictions.append(predicted_labels)
            targets.append(labels)

        # Convert predictions and targets to tensors
        pos_softmax_predictions = torch.stack(pos_softmax_predictions)
        targets = torch.stack(targets)

        # Gather across devices
        pos_softmax_predictions, targets = self.accelerator.gather_for_metrics((pos_softmax_predictions, targets))
        targets = targets.cpu().numpy()
        pos_softmax_predictions = pos_softmax_predictions.cpu().numpy()

        # Compute evaluation metrics
        if self.accelerator.is_main_process:
            metrics = calculate_metrics(pos_softmax_predictions, targets, PATHOLOGIES)
            self.metrics.append(metrics)
            save_metrics(self.metrics, PATHOLOGIES, self.results_folder)

            # Plot metrics
            plot_precision_recall_curve(targets, pos_softmax_predictions, PATHOLOGIES, self.results_folder)
            plot_roc_curve(targets, pos_softmax_predictions, PATHOLOGIES, self.results_folder)
            plot_per_class_f1(metrics, PATHOLOGIES, self.results_folder)

    def infer(self):
        """
        Runs distributed inference.
        """
        start_time = time.time()
        self.maybe_print("Evaluation started")

        if self.zero_shot:
            self.zeroshot()

        if self.visualize:
            self.vis.visualize(
                attention=False,
                grad_cam=True,
                occlusion=False
            )
            
        total_time = time.time() - start_time
        self.maybe_print(f"Evaluation completed. Total Evaluation Time: {str(timedelta(seconds=total_time))}")
