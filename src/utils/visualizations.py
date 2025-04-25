import time
import torch
import random
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import torch.distributed as dist
from collections import Counter
from datetime import timedelta

from scipy.ndimage import gaussian_filter
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, to_rgba, to_hex
from matplotlib.lines import Line2D

from models.ctclip import CTCLIP
from accelerate import Accelerator
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import BertTokenizer


# Set seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Force deterministic behavior
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


PATHOLOGIES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",
    "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
    "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",
    "Bronchiectasis", "Interlobular septal thickening"
]

COLORS = [
    "red", "green", "blue", "cyan", "magenta", "yellow",
    "orange", "purple", "pink", "lime",
    "teal", "brown", "olive", "navy", "gold", "salmon",
    "turquoise", "indigo"
]


def normalize(volume):
    volume = volume - volume.min()
    if volume.max() > 0:
        volume = volume / volume.max()
    return volume


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
        results_folder: Path,
        diff_embeds_folder: str,
        tokenizer: BertTokenizer
    ):
        self.model = model.module if hasattr(model, "module") else model
        self.accelerator = accelerator
        self.single_dataloader = single_dataloader
        self.dist_dataloader = dist_dataloader
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.results_folder = results_folder
        self.diff_embeds_folder = diff_embeds_folder

        self.maybe_print = print if self.accelerator.is_main_process else lambda *args, **kwargs: None
        self.saved_outputs = {}
        self.hooks = []

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

    
    def _save_activations_vq(self, module, input, output):
        features = output[0]
        self.saved_outputs["vq_features"] = features.detach()

        def save_grad(grad):
            self.saved_outputs["vq_gradients"] = grad

        features.register_hook(save_grad)


    def _save_activations_spatial(self, module, input, output):
        features = output[0]
        self.saved_outputs["spatial_features"] = features.detach()

        def save_grad(grad):
            self.saved_outputs["spatial_gradients"] = grad

        features.register_hook(save_grad)


    def _save_activations_temporal(self, module, input, output):
        features = output[0]
        self.saved_outputs["temporal_features"] = features.detach()

        def save_grad(grad):
            self.saved_outputs["temporal_gradients"] = grad

        features.register_hook(save_grad)

    
    def _register_hooks(self):
        """
        Create and register forward/backward hooks to capture targeted layer features and gradients.
        """
        target_layer = self.model.visual_transformer.vq
        hook = target_layer.register_forward_hook(self._save_activations_vq)
        self.hooks.append(hook)

        target_layer = self.model.visual_transformer.enc_spatial_transformer.layers[-1][1]
        hook = target_layer.register_forward_hook(self._save_activations_spatial)
        self.hooks.append(hook)

        target_layer = self.model.visual_transformer.enc_temporal_transformer.layers[-1][1]
        hook = target_layer.register_forward_hook(self._save_activations_temporal)
        self.hooks.append(hook)


    def _remove_hooks(self):
        """
        Remove all hooks attached to model from previous gradient-related activities.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    
    def _loss_function(self, sim_matrix, targets=None):
        """
        Compute contrastive loss using similarity matrix.
        """
        if targets is None:
            targets = torch.arange(sim_matrix.size(0), device=self.accelerator.device)

        loss_i2t = F.cross_entropy(sim_matrix, targets)
        loss_t2i = F.cross_entropy(sim_matrix.t(), targets)
        loss = (loss_i2t + loss_t2i) / 2

        return loss
    
    
    def visualize_overlay(self, image, overlay, scan_name, overlay_name, save_path, threshold=0.0, extra_info=""):
        """
        Creates and saves an animated visualization of a CT scan, an overlay and a combination of both.
        
        Args:
            image (np.array): 3D CT scan image [D, H, W].
            overlay (np.array): 3D normalized heatmap overlay [D, H, W].
            scan_name (str): Name of the scan.
            overlay_name (str): Name of the overlay.
            save_path (str): Path to save the animation.
        """
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


    def visualize_pathology_heatmaps(self, image, heatmaps, save_path, interval=100, figsize=(6, 6)):
        """
        Create an animated slice-by-slice overlay of heatmaps with a static legend.

        Parameters:
            image (ndarray): 3D array (Z, H, W) representing the background image (e.g., CT slices).
            heatmaps (dict): Dictionary mapping pathology names to 3D arrays (Z, H, W) of activation maps.
            interval (int): Delay between frames in milliseconds.
            figsize (tuple): Size of the matplotlib figure.
        
        Returns:
            ani (matplotlib.animation.ArtistAnimation): Animation object.
        """

        cmaps = {
            pathology: LinearSegmentedColormap.from_list(
                f"{pathology.replace(' ', '_')}_cmap",
                [to_rgba("black", 0.0), to_rgba(color, 1.0)]
            )
            for pathology, color in zip(PATHOLOGIES, COLORS)
        }

        pathology_colors = {
            pathology: to_hex(to_rgba(color, 1.0))
            for pathology, color in zip(PATHOLOGIES, COLORS)
        }

        fig, ax = plt.subplots(figsize=figsize)
        ims = []

        for slice_idx in range(image.shape[0]):
            im_frame = []

            # Base CT slice
            im = ax.imshow(image[slice_idx], cmap="bone", animated=True)
            im_frame.append(im)

            # Overlay each pathology's heatmap
            for pathology in heatmaps.keys():
                imslice = heatmaps[pathology][slice_idx]
                im2 = ax.imshow(imslice, cmap=cmaps[pathology], vmin=0, vmax=1, alpha=imslice, animated=True)
                im_frame.append(im2)

            ax.axis("off")
            ims.append(im_frame)

        legend_elements = [
            Line2D([0], [0], color=pathology_colors[pathology], lw=2, label=pathology)
            for pathology in heatmaps.keys()
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize='small', frameon=True)

        ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=False, repeat_delay=1000)
        ani.save(save_path, writer="pillow", fps=10)
        plt.close(fig)
    
    
    def visualize_integrated_gradients(self, image, text_tokens, labels, scan_name, original_scan_path, steps=50):
        blurred_np = gaussian_filter(image.squeeze().cpu().numpy(), sigma=5.0)
        baseline_image = torch.from_numpy(blurred_np).unsqueeze(0).unsqueeze(0).to(self.accelerator.device)
        self._register_hooks()

        _ = self.model(text_tokens, image)
        real_features = self.saved_outputs["vq_features"].view(-1, 512)

        _ = self.model(text_tokens, baseline_image)
        baseline_features = self.saved_outputs["vq_features"].view(-1, 512)

        grads = []
        image_diff = image - baseline_image

        for alpha in torch.linspace(0, 1, steps).to(self.accelerator.device):
            interpolated_input = baseline_image + alpha * image_diff
            interpolated_input.requires_grad = True

            # Forward + backward
            with torch.enable_grad():
                sim_matrix, *_ = self.model(text_tokens, interpolated_input)
                sim_matrix[self.rank, self.rank].backward()

            grads.append(self.saved_outputs["vq_gradients"].clone())

        self._remove_hooks()
        avg_gradients = torch.stack(grads).mean(dim=0)  # shape: [24*24*24, 512]
        integrated_grads = (real_features - baseline_features) * avg_gradients
        original_image = self._read_nii_data(original_scan_path).squeeze()

        cam = integrated_grads.sum(dim=-1).relu()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cam = cam.view(24, 24, 24)

        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                            size=original_image.shape,
                            mode='trilinear',
                            align_corners=False).squeeze(0).squeeze(0)
        cam = cam.detach().cpu().numpy()

        original_image = np.rot90(original_image, k=-1, axes=(1, 2))
        cam = np.rot90(cam, k=-1, axes=(1, 2))

        if self.accelerator.process_index == 0:
            results_dir = self._results_subdirectory("integrated_gradients")
            extra_info = "Text: original\nThreshold: 0.0\nIG steps: {}\nBaseline: -1".format(steps)
            self.visualize_overlay(original_image, cam, scan_name, "Integrated Gradients", results_dir / f"{scan_name}.gif", threshold=0.0, extra_info=extra_info)


    def visualize_grad_cam(self, image, text_tokens, labels, scan_name, original_scan_path):
        """
        Use spatial feature maps and gradients obtained from hooks to construct class activation maps (cams).
        Visualize these cams by overlaying them on top of the original CT scan.
        """
        self._register_hooks()
        with torch.enable_grad():
            sim_matrix, *_ = self.model(text_tokens, image)
            sim_matrix[self.rank, self.rank].backward()
        self._remove_hooks()

        original_image = self._read_nii_data(original_scan_path).squeeze()      # torch.Size([240, 480, 480])

        # spatial_features = self.saved_outputs.get("spatial_features").view(-1, 512)             # torch.Size([24, 576, 512]) --> torch.Size([13824, 512])
        # temporal_features = self.saved_outputs.get("temporal_features").view(-1, 512)             # torch.Size([24, 576, 512]) --> torch.Size([13824, 512])

        # spatial_gradients = self.saved_outputs.get("spatial_gradients").view(-1, 512)           # torch.Size([24, 576, 512]) --> torch.Size([13824, 512])
        # temporal_gradients = self.saved_outputs.get("temporal_gradients").view(-1, 512)           # torch.Size([24, 576, 512]) --> torch.Size([13824, 512])

        # spatial_gradients = spatial_gradients.mean(dim=0, keepdim=True)
        # temporal_gradients = temporal_gradients.mean(dim=0, keepdim=True)

        # spatial_cam = (spatial_gradients * spatial_features).sum(dim=-1).relu()
        # temporal_cam = (temporal_gradients * temporal_features).sum(dim=-1).relu()

        # spatial_cam = (spatial_cam - spatial_cam.min()) / (spatial_cam.max() + 1e-8)
        # temporal_cam = (temporal_cam - temporal_cam.min()) / (temporal_cam.max() + 1e-8)

        # cam = torch.sqrt(spatial_cam * temporal_cam + 1e-8)

        features = self.saved_outputs.get("vq_features").view(-1, 512)
        gradients = self.saved_outputs.get("vq_gradients").view(-1, 512).mean(dim=0, keepdim=True)
        
        cam = (gradients * features).sum(dim=-1)
        cam = cam.relu()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cam = cam.view(24, 24, 24)

        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                            size=original_image.shape,
                            mode='trilinear',
                            align_corners=False).squeeze(0).squeeze(0)
        cam = cam.detach().cpu().numpy()

        original_image = np.rot90(original_image, k=-1, axes=(1, 2))
        cam = np.rot90(cam, k=-1, axes=(1, 2))

        if self.accelerator.process_index == 0:
            results_dir = self._results_subdirectory("grad_cam")
            extra_info = "Text: original\nThreshold: 0.0\nBackprop: rank sim score"
            self.visualize_overlay(original_image, cam, scan_name, "Grad-CAM", results_dir / f"{scan_name}.gif", threshold=0.0, extra_info=extra_info)


    def visualize_occlusion_sensitivity(self, image, text_tokens, labels, scan_name, original_scan_path, patch_size=(20,40,40), stride=(20,40,40)):
        arithmetic_embeds = np.load(self.diff_embeds_folder, allow_pickle=True)
        arithmetic_embeds = arithmetic_embeds.item()
        tensor_embeds = {k: torch.tensor(v, dtype=torch.float32).to(self.accelerator.device).unsqueeze(0) for k, v in arithmetic_embeds.items()}

        original_image = self._read_nii_data(original_scan_path)
        _, _, D, H, W = image.shape
        threshold = 0.5

        # Divide Depth (D) among GPUs
        # Each GPU is responsible for a subset of the depth slices (D)
        # This part is needed if occlusion is run with multiple GPU's
        d_coords = range(0, D - patch_size[0] + 1, stride[0])
        h_coords = range(0, H - patch_size[1] + 1, stride[1])
        w_coords = range(0, W - patch_size[2] + 1, stride[2])

        all_patch_coords = [
            (d, h, w)
            for d in d_coords
            for h in h_coords
            for w in w_coords
        ]

        total_patches = len(all_patch_coords)
        patches_per_rank = total_patches // self.world_size
        trimmed_total = patches_per_rank * self.world_size

        # Drop extra patches if needed
        all_patch_coords = all_patch_coords[:trimmed_total]

        # Now split evenly
        start = self.rank * patches_per_rank
        end = start + patches_per_rank
        patch_coords = all_patch_coords[start:end]

        total_patches = len(patch_coords)
        print(f"[Rank {self.rank}] Total patches to go through: {total_patches}")

        start_time = time.time()

        positive_indices = (labels == 1).nonzero(as_tuple=True)[0]
        positive_pathologies = [PATHOLOGIES[i] for i in positive_indices.tolist()]
        heatmaps = {}

        for pos_pathology in positive_pathologies:
            self.maybe_print("Processing pathology:", pos_pathology)
            text_embeds = tensor_embeds[pos_pathology]
            heatmap = np.zeros((D, H, W))
            count_map = np.zeros((D, H, W))

            # Compute original similarity score (before occlusion)
            with torch.no_grad():
                original_sim_matrix, *_ = self.model(None, image, text_embeds)
                original_score = original_sim_matrix[self.rank, self.rank].item()

            # Iterate through the assigned patch coordinates and occlude voxel windows
            for idx, (d, h, w) in enumerate(patch_coords):
                occluded_image = image.clone().detach()
                occluded_image[:, :, d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] = -1

                occluded_sim_matrix, *_ = self.model(None, occluded_image, text_embeds)
                occluded_score = occluded_sim_matrix[self.rank, self.rank].item()

                importance = max(original_score - occluded_score, 0)
                heatmap[d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] += importance
                count_map[d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] += 1

                if idx % 100 == 0 or idx == total_patches - 1:
                    elapsed = time.time() - start_time
                    avg_time_per_patch = elapsed / (idx + 1)
                    remaining_time = avg_time_per_patch * (total_patches - (idx + 1))
                    percent_done = 100.0 * (idx + 1) / total_patches

                    print(f"[Rank {self.rank}] Patch {idx + 1}/{total_patches} "
                        f"({percent_done:.2f}%) - Elapsed: {elapsed:.1f}s - ETA: {remaining_time:.1f}s")


            heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32, device=self.accelerator.device)
            count_tensor = torch.tensor(count_map, dtype=torch.float32, device=self.accelerator.device)

            if self.world_size > 1:
                dist.reduce(heatmap_tensor, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(count_tensor, dst=0, op=dist.ReduceOp.SUM)

            if self.accelerator.is_main_process:
                count_tensor[count_tensor == 0] = 1
                heatmap_tensor = heatmap_tensor / count_tensor
                heatmap_tensor = (heatmap_tensor - heatmap_tensor.min()) / (heatmap_tensor.max() - heatmap_tensor.min() + 1e-8)

                full_heatmap = heatmap_tensor.unsqueeze(0).unsqueeze(0)
                full_heatmap = F.interpolate(full_heatmap,
                                size=original_image.shape,
                                mode='trilinear',
                                align_corners=False).squeeze(0).squeeze(0)
                full_heatmap = full_heatmap.detach().cpu().numpy()
                full_heatmap[full_heatmap < threshold] = 0
                full_heatmap = np.rot90(full_heatmap, k=-1, axes=(1, 2))
                heatmaps[pos_pathology] = full_heatmap

        if self.accelerator.is_main_process:
            extra_info = f"""
                Text: embed diff
                Threshold: {threshold}
                Patch size: {patch_size}
                Stride: {stride}
            """
            results_dir = self._results_subdirectory("occlusion")
            np.save(results_dir / f"{scan_name}_{str(patch_size)}_{str(stride)}_heatmaps.npy", heatmaps)
            # self.visualize_overlay(original_image, full_heatmap, scan_name, "Occlusion", results_dir / f"{scan_name}.gif", extra_info=extra_info)
            original_image = np.rot90(original_image, k=-1, axes=(1, 2))
            self.visualize_pathology_heatmaps(original_image, heatmaps, results_dir / f"{scan_name}_{str(patch_size)}_{str(stride)}_occlusion.gif")


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
                visual_func = self.visualize_integrated_gradients
            elif name == "grad_cam":
                dataloader = self.dist_dataloader
                visual_func = self.visualize_grad_cam
            elif name == "occlusion":
                dataloader = self.single_dataloader
                visual_func = self.visualize_occlusion_sensitivity
            else:
                self.maybe_print(f"{name} is not correct visualization argument, returning.")
                return

            start_time = time.time()
            self.maybe_print(f"{name} visualization started.")

            for batch in iter(dataloader):
                image, texts, labels, scan_names, original_scan_paths = [b.to(self.accelerator.device) if isinstance(b, torch.Tensor) else b for b in batch]
                text_tokens = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                ).to(self.accelerator.device)

                self.model.zero_grad()
                visual_func(image, text_tokens, labels[0], scan_names[0], original_scan_paths[0])

            total_time = time.time() - start_time
            self.maybe_print(f"{name} visualization completed. Total visualization time: {str(timedelta(seconds=total_time))}")