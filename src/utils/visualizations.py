import time
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import torch.distributed as dist
from collections import Counter
from datetime import timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats as stats
from mpl_toolkits.axes_grid1 import make_axes_locatable

from CTCLIP import CTCLIP
from accelerate import Accelerator
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import BertTokenizer


PATHOLOGIES = [
            "Medical material", "Arterial wall calcification", "Cardiomegaly",
            "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
            "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",
            "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
            "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",
            "Bronchiectasis", "Interlobular septal thickening"
        ]


class Visualizations():
    """
    Visualizations class for approaches like:
        - Attention Maps
        - Grad-CAM
        - Occlusion Sensitivity
    """
    def __init__(
        self,
        model: CTCLIP,
        accelerator: Accelerator,
        single_dataloader: DataLoader,
        dist_dataloader: DataLoader,
        batch_size: int,
        tokenizer: BertTokenizer,
        results_folder: Path
    ):
        self.model = model.module if hasattr(model, "module") else model
        self.accelerator = accelerator
        self.single_dataloader = single_dataloader
        self.dist_dataloader = dist_dataloader
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.results_folder = results_folder

        self.maybe_print = print if self.accelerator.is_main_process else lambda *args, **kwargs: None
        self.saved_outputs = {}

        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes


    def _results_subdirectory(self, visualization_name):
        """
        Create subdirectory based on visualization name.
        Construct and return Path via dynamic indexing.
        """
        results_subdir = self.results_folder / visualization_name
        results_subdir.mkdir(parents=True, exist_ok=True)

        existing_folders = [d for d in results_subdir.iterdir() if d.is_dir()]
        idx = len(existing_folders) + 1

        results_subdir = results_subdir / str(idx)
        results_subdir.mkdir(parents=True, exist_ok=True)

        return results_subdir


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

    
    def _register_hooks(self):
        """
        Create and register forward/backward hooks to capture spatial features and gradients.
        """
        spatial_attention_layer = self.model.visual_transformer.enc_spatial_transformer

        spatial_attention_layer.register_forward_hook(
            lambda module, input, output: self.saved_outputs.update({"spatial_features": output[0]})
        )

        spatial_attention_layer.register_full_backward_hook(
            lambda module, grad_input, grad_output: self.saved_outputs.update({"spatial_gradients": grad_output[0]})
        )

    
    def _loss_function(self, sim_matrix, targets=None):
        """
        Compute contrastive loss using similarity matrix.
        """
        if targets is None:
            targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        loss_i2t = F.cross_entropy(sim_matrix, targets)
        loss_t2i = F.cross_entropy(sim_matrix.t(), targets)
        loss = (loss_i2t + loss_t2i) / 2

        return loss


    def visualize_overlay(self, image, overlay, scan_name, overlay_name, save_path, threshold=0.0, extra_info=""):
        """
        Creates and saves an animated visualization of a CT scan, an overlay and a combination of both.
        
        Args:
            image (np.array): 3D CT scan image [D, H, W].
            overlay (np.array): 3D heatmap overlay [D, H, W].
            scan_name (str): Name of the scan.
            overlay_name (str): Name of the overlay.
            save_path (str): Path to save the animation.
        """
        # Normalize overlay between 0 and 1 for visualization
        overlay = (overlay - np.min(overlay)) / (np.max(overlay) - np.min(overlay) + 1e-8)
        overlay[overlay < threshold] = 0

        # Set up figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Scan: {scan_name}", fontsize=16)
        ims = []

        # Add extra text at the top-left corner of the figure
        if extra_info:
            fig.text(0.01, 0.98, extra_info, fontsize=12, ha="left", va="top", bbox=dict(facecolor="white", alpha=0.5, edgecolor="black"))

        # Loop over slices to create animation
        for slice_idx in range(image.shape[0]):
            # First subplot: Original CT Scan
            im1 = axes[0].imshow(image[slice_idx, :, :], cmap="bone", animated=True)
            axes[0].set_title("Original Scan", fontsize=12)
            axes[0].axis("off")

            # Second subplot: Overlay
            im2 = axes[1].imshow(overlay[slice_idx, :, :], cmap="inferno", vmin=0, vmax=1, animated=True)
            axes[1].set_title(f"{overlay_name} Heatmap", fontsize=12)
            axes[1].axis("off")

            # Third subplot: CT Scan + Overlay
            im3 = axes[2].imshow(image[slice_idx, :, :], cmap="bone", animated=True)
            im4 = axes[2].imshow(overlay[slice_idx, :, :], cmap="inferno", alpha=0.6, vmin=0, vmax=1, animated=True)
            axes[2].set_title("Scan + Heatmap", fontsize=12)
            axes[2].axis("off")

            ims.append([im1, im2, im3, im4])

        cbar_ax = fig.add_axes([0.35, 0.08, 0.3, 0.02])  # [left, bottom, width, height]
        cbar = fig.colorbar(im2, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(f"{overlay_name} Intensity", fontsize=12)

        # Create and save animation
        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False, repeat_delay=1000)
        ani.save(save_path, writer="pillow", fps=10)
        plt.close(fig)


    def visualize_attention(self, image, text_tokens, scan_name, original_scan_path):
        """
        original_image torch.Size([1, 1, 240, 480, 480])
        image_tokens torch.Size([1, 24, 24, 24, 512])
        attention_weights torch.Size([24, 8, 576, 576])
        """
        with torch.no_grad():
            *_, image_tokens, attention_weights = self.model(text_tokens, image)
        
        original_image = self._read_nii_data(original_scan_path)
        image_tokens = image_tokens.norm(dim=-1)           # L2 norm over features

        attention_weights = attention_weights.mean(dim=1)  # Average over heads
        attention_weights = attention_weights.sum(dim=-1)  # Sum over patches
        attention_weights = attention_weights.view(-1, 24, 24, 24)

        importance_map = image_tokens * attention_weights
        importance_map = F.interpolate(importance_map.unsqueeze(0),
                            size=original_image.shape,
                            mode='trilinear',
                            align_corners=False).squeeze(0).squeeze(0)
        importance_map = importance_map.detach().cpu().numpy()

        if self.accelerator.is_main_process:
            results_dir = self._results_subdirectory("attention") / f"{scan_name}.gif"
            extra_info = "Threshold: 0.8"
            self.visualize_overlay(original_image, importance_map, scan_name, "Attention", results_dir, threshold=0.8, extra_info=extra_info)


    def visualize_grad_cam(self, image, text_tokens, scan_name, original_scan_path):
        """
        Use spatial feature maps and gradients obtained from hooks to construct class activation maps (cams).
        Visualize these cams by overlaying them on top of the original CT scan.
        """
        with torch.enable_grad():
            sim_matrix, *_ = self.model(text_tokens, image)
            sim_matrix[self.rank, self.rank].backward()

        original_image = self._read_nii_data(original_scan_path).squeeze(0)                                                     # torch.Size([1, 1, 240, 480, 480])
        spatial_features = self.saved_outputs.get("spatial_features")                                                           # torch.Size([24, 576, 512])
        spatial_gradients = self.saved_outputs.get("spatial_gradients")                                                         # torch.Size([24, 576, 512])
        
        spatial_features = spatial_features.to(torch.float64)
        spatial_gradients = spatial_gradients.to(torch.float64)

        spatial_gradients = spatial_gradients.mean(dim=1, keepdim=True)
        
        cam = (spatial_gradients * spatial_features).sum(dim=-1)
        cam = cam.relu()
        cam = cam.view(24, 24, 24)

        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                            size=original_image.shape,
                            mode='trilinear',
                            align_corners=False).squeeze(0).squeeze(0)
        cam = cam.detach().cpu().numpy()

        if self.accelerator.process_index == 0:
            results_dir = self._results_subdirectory("grad_cam")
            extra_info = "Text: original\nThreshold: 0.0\nBackprop: rank sim score\nGrad target: input"
            self.visualize_overlay(original_image, cam, scan_name, "Grad-CAM", results_dir / f"{scan_name}.gif", threshold=0.0, extra_info=extra_info)


    def visualize_occlusion_sensitivity(self, image, text_tokens, scan_name, original_scan_path, patch_size=(40,80,80), stride=(40,80,80)):
        black_value = image.max().item()
        original_image = self._read_nii_data(original_scan_path)
        _, _, D, H, W = image.shape
        heatmap = np.zeros((D, H, W))
        importance_values = []
        iteration_count = 0

        # Compute original similarity score (before occlusion)
        with torch.no_grad():
            original_sim_matrix, *_ = self.model(text_tokens, image)
            original_score = original_sim_matrix[self.rank, self.rank].item()

        # Divide Depth (D) among GPUs
        # Each GPU is responsible for a subset of the depth slices (D)
        depth_splits = torch.chunk(torch.arange(D), self.world_size)
        d_start, d_end = depth_splits[self.rank][0].item(), min(depth_splits[self.rank][-1].item() + 1, D)

        # Iterate through the assigned depth range for this GPU
        for d in range(d_start, min(d_end, D - patch_size[0] + 1), stride[0]):
            for h in range(0, min(H, H - patch_size[1] + 1), stride[1]):
                for w in range(0, min(W, W - patch_size[2] + 1), stride[2]):
                    occluded_image = image.clone().detach()
                    occluded_image[:, :, d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] = black_value

                    with torch.no_grad():
                        occluded_sim_matrix, *_ = self.model(text_tokens, occluded_image)
                        occluded_score = occluded_sim_matrix[self.rank, self.rank].item()

                    # Compute importance based on drop in diagonal sum
                    importance = original_score - occluded_score
                    self.maybe_print("Importance:", importance)

                    # Update heatmap
                    heatmap[d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] = importance
        
        self.maybe_print("heatmap first 10 values before", heatmap[0, 0, :10])
        median_value = np.mean(heatmap[0, 0, 10:30])
        heatmap[0, 0, :10] = median_value
        self.maybe_print("heatmap first 10 values after", heatmap[0, 0, :10])

        heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32, device=self.accelerator.device)
        heatmap_tensor = (heatmap_tensor - heatmap_tensor.min()) / (heatmap_tensor.max() - heatmap_tensor.min() + 1e-8)

        if self.world_size > 1:
            dist.reduce(heatmap_tensor, dst=0, op=dist.ReduceOp.SUM)

        if self.accelerator.is_main_process:            
            full_heatmap = heatmap_tensor.unsqueeze(0).unsqueeze(0)
            full_heatmap = F.interpolate(full_heatmap,
                            size=original_image.shape,
                            mode='trilinear',
                            align_corners=False).squeeze(0).squeeze(0)
            full_heatmap = full_heatmap.detach().cpu().numpy()

            results_dir = self._results_subdirectory("occlusion")
            extra_info = f"Text: original\nThreshold: 0.0\nPatch size: {patch_size}\nStride: {stride}"
            self.visualize_overlay(original_image, full_heatmap, scan_name, "Occlusion", results_dir / f"{scan_name}.gif", extra_info=extra_info)


    def visualize(self, **kwargs):
        """
        Handle visualization controls. Expects keyword arguments.
        """
        for name, val in kwargs.items():
            if not val:
                continue

            if self.world_size > 1:
                torch.distributed.barrier()  # Ensure all processes are at the same point
            
            if name == "attention":
                dataloader = self.dist_dataloader
                visual_func = self.visualize_attention
            elif name == "grad_cam":
                dataloader = self.dist_dataloader
                visual_func = self.visualize_grad_cam
                self._register_hooks()
            elif name == "occlusion":
                dataloader = self.single_dataloader
                visual_func = self.visualize_occlusion_sensitivity
            else:
                self.maybe_print(f"{name} is not correct visualization argument, returning.")
                return

            start_time = time.time()
            self.maybe_print(f"{name} visualization started.")

            for batch in iter(dataloader):
                image, texts, _, scan_names, original_scan_paths = [b.to(self.accelerator.device) if isinstance(b, torch.Tensor) else b for b in batch]
                text_tokens = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                ).to(self.accelerator.device)

                self.model.zero_grad()
                visual_func(image, text_tokens, scan_names[0], original_scan_paths[0])

            total_time = time.time() - start_time
            self.maybe_print(f"{name} visualization completed. Total visualization time: {str(timedelta(seconds=total_time))}")