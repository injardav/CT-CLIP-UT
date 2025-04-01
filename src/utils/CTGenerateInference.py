import time
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path
from datetime import timedelta, datetime
from scipy.ndimage import zoom

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from accelerate.utils import DistributedDataParallelKwargs

from utils.InferenceDataset import InferenceDataset
from models.ctgenerate import CTGENERATE


PATHOLOGIES = [
            "Medical material", "Arterial wall calcification", "Cardiomegaly",
            "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
            "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",
            "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
            "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",
            "Bronchiectasis", "Interlobular septal thickening"
        ]


class CTGenerateInference(nn.Module):
    """
    CTGenerateInference for evaluation of GENERATE model.
    """
    def __init__(
        self,
        model: CTGENERATE,
        batch_size: int,
        data_valid: str,
        valid_reports: str,
        valid_labels: str,
        valid_metadata: str,
        results_folder: str = "./results",
        num_workers: int = 8,
        num_valid_samples: int = 0
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

        self.num_valid_samples = num_valid_samples if num_valid_samples else (1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())
        self.batch_size = batch_size
        self.ds = InferenceDataset(data_folder=data_valid, reports=valid_reports, metadata=valid_metadata, labels=valid_labels, num_samples=self.num_valid_samples, model_type="ctgenerate")

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

        self.dl = DataLoader(self.ds, batch_size=batch_size, sampler=self.sampler, num_workers=num_workers)

        (
            self.model
        ) = self.accelerator.prepare(
            self.model
        )

        self.results_folder = results_folder
        if self.accelerator.is_main_process:
            # Create correct results directory structure with date
            self.results_folder = Path(results_folder) / datetime.now().strftime("%d-%m-%Y")
            self.results_folder.mkdir(parents=True, exist_ok=True)
        
        self.maybe_print(f"Validation size: {len(self.dl.dataset)}")

    def _read_nii_data(self, file_path):
        """
        Load the NIfTI file using nibabel.
        """
        try:
            nii_img = nib.load(file_path)
            nii_data = nii_img.get_fdata()
            nii_data = nii_data.transpose(2, 0, 1)  # Transpose to (D, H, W)
            tensor_data = torch.from_numpy(nii_data).float()
            return tensor_data
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return

    def infer(self):
        """
        Runs distributed inference.
        """
        start_time = time.time()
        self.maybe_print("CTGENERATE inference started")
    
        self.model.eval()
        for batch in iter(self.dl):
            with torch.no_grad():
                image, text, labels, scan_name, original_scan_path = [b.to(self.accelerator.device) if isinstance(b, torch.Tensor) else b for b in batch]
                positive_pathologies = [p for p, l in zip(PATHOLOGIES, labels[0].tolist()) if l == 1.0]
                
                feature_map, pathology_attention = self.model(image, text, positive_pathologies)
            
            original_image = torch.Tensor(self._read_nii_data(original_scan_path[0])).unsqueeze(0).unsqueeze(0)
            original_image = F.interpolate(original_image, size=image.shape[2:], mode="trilinear", align_corners=False).squeeze().numpy()

            for pathology, cross_attention in pathology_attention.items():
                print(f"Computing cross_attention for {scan_name} with pathology: {pathology}")
                print("cross_attention shape", cross_attention.shape)
                cross_attention_weights = cross_attention.mean(dim=1).mean(dim=-1)  # Average over heads and pathology tokens
                cross_attention_weights = cross_attention_weights.view(1, 101, 8, 8).unsqueeze(0)
                cross_attention_weights = F.interpolate(cross_attention_weights, size=image.shape[2:], mode="trilinear", align_corners=False)
                cross_attention_weights = cross_attention_weights.squeeze().detach().cpu().numpy()
                cross_attention_weights = np.rot90(cross_attention_weights, k=-1, axes=(1, 2))
                cross_attention_weights = (cross_attention_weights - np.min(cross_attention_weights)) / (np.max(cross_attention_weights) - np.min(cross_attention_weights) + 1e-8)
    
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f"Scan: {scan_name[0]} | Pathology: {pathology}", fontsize=16)
                ims = []
    
                original_image = np.rot90(original_image, k=-1, axes=(1, 2))
                for slice_idx in range(original_image.shape[0]):
                    im1 = axes[0].imshow(original_image[slice_idx, :, :], cmap="bone", animated=True)
                    axes[0].set_title("Original Scan", fontsize=12)
                    axes[0].axis("off")
    
                    im2 = axes[1].imshow(cross_attention_weights[slice_idx, :, :], cmap="inferno", vmin=0, vmax=1, animated=True)
                    axes[1].set_title(f"GenerateCT Attention Heatmap", fontsize=12)
                    axes[1].axis("off")
    
                    im3 = axes[2].imshow(original_image[slice_idx, :, :], cmap="bone", animated=True)
                    im4 = axes[2].imshow(cross_attention_weights[slice_idx, :, :], cmap="inferno", alpha=0.6, vmin=0, vmax=1, animated=True)
                    axes[2].set_title("Scan + Heatmap", fontsize=12)
                    axes[2].axis("off")
    
                    ims.append([im1, im2, im3, im4])
    
                cbar_ax = fig.add_axes([0.35, 0.08, 0.3, 0.02])  # [left, bottom, width, height]
                cbar = fig.colorbar(im2, cax=cbar_ax, orientation="horizontal")
                cbar.set_label(f"GenerateCT Attention Intensity", fontsize=12)
    
                # Create and save animation
                ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False, repeat_delay=1000)
                ani.save(self.results_folder / f"ctgenerate_{scan_name[0]}_{pathology}.gif", writer="pillow", fps=10)
                plt.close(fig)

        total_time = time.time() - start_time
        self.maybe_print(f"CTGENERATE inference completed. Total inference Time: {str(timedelta(seconds=total_time))}")
