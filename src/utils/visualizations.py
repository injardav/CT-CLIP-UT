import gc
import time
import torch
import random
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import torch.distributed as dist
from collections import Counter
from datetime import timedelta
from tqdm import tqdm

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

from torch.utils.data import DataLoader, Dataset
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

SEGMENTABLE_TERMS = [
    "lymph nodes", "pleural effusion", "ground glass",
    "lung parenchyma", "right lobe", "left lobe", "upper lobe",
    "lower lobe", "mediastinal mass", "lung nodules", "bone lesion",
    "right lung", "left lung", "abdominal organs"
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
        dataset: Dataset,
        dist_dataloader: DataLoader,
        batch_size: int,
        results_folder: Path,
        diff_embeds_folder: str,
        tokenizer: BertTokenizer
    ):
        self.model = model.module if hasattr(model, "module") else model
        self.accelerator = accelerator
        self.dataset = dataset
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
        """
        Hook to capture VQ features and gradients.
        """
        features = output[0]
        self.saved_outputs["vq_features"] = features.detach()

        def save_grad(grad):
            self.saved_outputs["vq_gradients"] = grad

        features.register_hook(save_grad)


    def _save_spatial_layer_outputs(self, module, input, output):
        """
        Hook for spatial self-attention layer. Captures:
        - Feature map (output[0])
        - Attention weights (output[1])
        - Gradients of the feature map
        """
        feature_map, attn_weights = output

        self.saved_outputs["spatial_features"].append(feature_map.detach())
        self.saved_outputs["spatial_attention_weights"].append(attn_weights.detach())

        def save_grad(grad):
            self.saved_outputs["spatial_gradients"].append(grad)

        feature_map.register_hook(save_grad)


    def _save_temporal_layer_outputs(self, module, input, output):
        """
        Hook for temporal self-attention layer. Captures:
        - Feature map (output[0])
        - Attention weights (output[1])
        - Gradients of the feature map
        """
        feature_map, attn_weights = output

        self.saved_outputs["temporal_features"].append(feature_map.detach())
        self.saved_outputs["temporal_attention_weights"].append(attn_weights.detach())

        def save_grad(grad):
            self.saved_outputs["temporal_gradients"].append(grad)

        feature_map.register_hook(save_grad)


    def _save_spatial_ff_outputs(self, module, input, output):
        """
        Hook for spatial FeedForward layer. Captures:
        - Feature map
        - Gradients of the feature map
        """
        feature_map = output  # output is a single tensor from nn.Sequential

        self.saved_outputs["spatial_ff_features"].append(feature_map.detach())

        def save_grad(grad):
            self.saved_outputs["spatial_ff_gradients"].append(grad)

        feature_map.register_hook(save_grad)


    def _save_temporal_ff_outputs(self, module, input, output):
        """
        Hook for spatial FeedForward layer. Captures:
        - Feature map
        - Gradients of the feature map
        """
        feature_map = output  # output is a single tensor from nn.Sequential

        self.saved_outputs["temporal_ff_features"].append(feature_map.detach())

        def save_grad(grad):
            self.saved_outputs["temporal_ff_gradients"].append(grad)

        feature_map.register_hook(save_grad)

    
    def _register_hooks(self):
        """
        Register forward hooks to capture:
        - VQ features and gradients
        - Spatial and temporal attention features, attention weights, and gradients from all layers
        """
        # Clear outputs
        self.saved_outputs["vq_features"] = []
        self.saved_outputs["spatial_attention_weights"] = []
        self.saved_outputs["temporal_attention_weights"] = []
        self.saved_outputs["spatial_features"] = []
        self.saved_outputs["temporal_features"] = []
        self.saved_outputs["vq_gradients"] = []
        self.saved_outputs["spatial_gradients"] = []
        self.saved_outputs["temporal_gradients"] = []
        self.saved_outputs["spatial_ff_features"] = []
        self.saved_outputs["spatial_ff_gradients"] = []
        self.saved_outputs["temporal_ff_features"] = []
        self.saved_outputs["temporal_ff_gradients"] = []

        # VQ features and gradients
        vq_layer = self.model.visual_transformer.vq
        self.hooks.append(vq_layer.register_forward_hook(self._save_activations_vq))

        # Register hooks for all spatial self-attention layers
        for layer in self.model.visual_transformer.enc_spatial_transformer.layers:
            attn_module = layer[1]
            self.hooks.append(attn_module.register_forward_hook(self._save_spatial_layer_outputs))

        # Register hooks for all temporal self-attention layers
        for layer in self.model.visual_transformer.enc_temporal_transformer.layers:
            attn_module = layer[1]
            self.hooks.append(attn_module.register_forward_hook(self._save_temporal_layer_outputs))

        # Register hooks for all spatial feedforward layers
        for layer in self.model.visual_transformer.enc_spatial_transformer.layers:
            ff_module = layer[3]
            self.hooks.append(ff_module.register_forward_hook(self._save_spatial_ff_outputs))

        # Register hooks for all temporal feedforward layers
        for layer in self.model.visual_transformer.enc_temporal_transformer.layers:
            ff_module = layer[3]
            self.hooks.append(ff_module.register_forward_hook(self._save_temporal_ff_outputs))


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


    def _upsample(self, input, target_shape):
        """
        Apply interpolation to a 3D tensor input.
        """
        return F.interpolate(input.unsqueeze(0).unsqueeze(0), size=target_shape, mode="trilinear", align_corners=False).squeeze().detach().cpu().to(torch.float32).numpy()

    
    def _broadcast_sample(self, sample, src=0):
        if self.rank == src:
            sample_list = [x.unsqueeze(0).to(self.accelerator.device) if isinstance(x, torch.Tensor) else [x] for x in sample]
        else:
            # Probe structure from rank 0's sample
            with torch.no_grad():
                probe_sample = self.dataset[0]
                sample_list = []
                for x in probe_sample:
                    if isinstance(x, torch.Tensor):
                        empty_tensor = torch.empty((1, *x.shape), dtype=x.dtype, device=self.accelerator.device)
                        sample_list.append(empty_tensor)
                    else:
                        sample_list.append([None])

        for i in range(len(sample_list)):
            if isinstance(sample_list[i], torch.Tensor):
                sample_list[i] = sample_list[i].contiguous()
                torch.distributed.broadcast(sample_list[i], src=src)
            else:
                torch.distributed.broadcast_object_list(sample_list[i], src=src)

        return sample_list

    
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


    def _compute_occlusion(self, image, text_tokens, text_embeds, patch_size, stride, threshold):
        # Divide Depth (D) among GPUs
        # Each GPU is responsible for a subset of the depth slices (D)
        # This part is needed if occlusion is run with multiple GPU's
        _, _, D, H, W = image.shape
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

        heatmap = np.zeros((D, H, W))
        count_map = np.zeros((D, H, W))

        # Compute original similarity score (before occlusion)
        with torch.no_grad():
            if isinstance(text_embeds, torch.Tensor) and text_embeds.ndim > 1:
                original_sim_matrix, *_ = self.model(None, image, text_embeds)
            else:
                original_sim_matrix, *_ = self.model(text_tokens, image)
            original_score = original_sim_matrix[self.rank, self.rank].item()

        # Iterate through the assigned patch coordinates and occlude voxel windows
        start_time = time.time()
        for idx, (d, h, w) in enumerate(patch_coords):
            occluded_image = image.clone().detach()
            occluded_image[:, :, d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] = -1

            if isinstance(text_embeds, torch.Tensor) and text_embeds.ndim > 1:
                occluded_sim_matrix, *_ = self.model(None, occluded_image, text_embeds)
            else:
                occluded_sim_matrix, *_ = self.model(text_tokens, occluded_image)

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
                            size=(D, H, W),
                            mode='trilinear',
                            align_corners=False).squeeze(0).squeeze(0)
            full_heatmap = full_heatmap.detach().cpu().numpy()
            full_heatmap[full_heatmap < threshold] = 0
            full_heatmap = np.rot90(full_heatmap, k=-1, axes=(1, 2))
            return full_heatmap
    
    
    def visualize_overlay(self, image, overlay, scan_name, overlay_name, save_path,
                      threshold=0.0, extra_info="", display_flags=None):
        """
        Creates and saves an animated visualization of a CT scan with selectable views.
        
        Args:
            image (np.array): 3D CT scan image [D, H, W].
            overlay (np.array): 3D normalized heatmap overlay [D, H, W].
            scan_name (str): Name of the scan.
            overlay_name (str): Name of the overlay.
            save_path (str): Path to save the animation.
            threshold (float): Minimum value in overlay to be visualized.
            extra_info (str): Extra text to display on the figure.
            display_flags (dict): Keys = {"original", "heatmap", "overlay"}, values = bool.
        """
        if display_flags is None:
            display_flags = {"original": True, "heatmap": True, "overlay": True}

        overlay = np.copy(overlay)
        overlay[overlay < threshold] = 0

        view_order = []
        if display_flags.get("original"): view_order.append("original")
        if display_flags.get("heatmap"):  view_order.append("heatmap")
        if display_flags.get("overlay"):  view_order.append("overlay")

        fig, axes = plt.subplots(1, len(view_order), figsize=(6 * len(view_order), 6))
        if len(view_order) == 1:
            axes = [axes]

        fig.suptitle(f"Scan: {scan_name}", fontsize=16)
        ims = []

        if extra_info:
            fig.text(0.00, 0.99, extra_info, fontsize=10, ha="left", va="top")

        for slice_idx in range(image.shape[0]):
            frame_images = []

            for ax, view in zip(axes, view_order):
                if view == "original":
                    im = ax.imshow(image[slice_idx], cmap="bone", animated=True)
                    ax.set_title("Original Scan", fontsize=12)
                    frame_images.append(im)
                elif view == "heatmap":
                    im = ax.imshow(overlay[slice_idx], cmap="inferno", vmin=0, vmax=1, animated=True)
                    ax.set_title(f"{overlay_name} Heatmap", fontsize=12)
                    frame_images.append(im)
                elif view == "overlay":
                    im1 = ax.imshow(image[slice_idx], cmap="bone", animated=True)
                    im2 = ax.imshow(overlay[slice_idx], cmap="inferno", alpha=overlay[slice_idx], vmin=0, vmax=1, animated=True)
                    ax.set_title("Scan + Heatmap", fontsize=12)
                    frame_images.extend([im1, im2])
                ax.axis("off")

            ims.append(frame_images)

        if "heatmap" in view_order:
            cbar_ax = fig.add_axes([0.35, 0.08, 0.3, 0.02])
            cbar = fig.colorbar(ims[0][view_order.index("heatmap")], cax=cbar_ax, orientation="horizontal")
            cbar.set_label(f"{overlay_name} Intensity", fontsize=12)

        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False, repeat_delay=1000)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ani.save(save_path, writer="pillow", fps=10)
        plt.close(fig)


    def visualize_pathology_heatmaps(self, image, heatmaps, save_path, interval=100, figsize=None):
        """
        Create an animated slice-by-slice grid:
        For each pathology, display a row with:
        [Original scan | Heatmap only | Scan + Heatmap overlay]

        Parameters:
            image (ndarray): 3D array (Z, H, W) representing the background image (e.g., CT slices).
            heatmaps (dict): Dictionary mapping pathology names to 3D arrays (Z, H, W) of activation maps.
            interval (int): Delay between frames in milliseconds.
            figsize (tuple): Size of the matplotlib figure.

        Returns:
            ani (matplotlib.animation.ArtistAnimation): Animation object.
        """

        if figsize is None:
            figsize = (12, 4 * len(heatmaps))

        cmaps = {
            pathology: LinearSegmentedColormap.from_list(
                f"{pathology.replace(' ', '_')}_cmap",
                [to_rgba("black", 0.0), to_rgba(color, 1.0)]
            )
            for pathology, color in zip(PATHOLOGIES, COLORS)
        }

        fig, axes = plt.subplots(nrows=len(heatmaps), ncols=3, figsize=figsize)
        if len(heatmaps) == 1:
            axes = np.expand_dims(axes, axis=0)

        ims = []

        num_slices = image.shape[0]
        pbar = tqdm(total=num_slices, desc="Generating frames", unit="slice", mininterval=10.0)

        start_time = time.time()
        for slice_idx in range(num_slices):
            im_frame = []

            for row_idx, (pathology, heatmap) in enumerate(heatmaps.items()):
                slice_img = image[slice_idx]
                slice_heatmap = heatmap[slice_idx]

                # Original scan
                im1 = axes[row_idx, 0].imshow(slice_img, cmap="bone", animated=True)
                axes[row_idx, 0].set_title(f"{pathology} - Scan", fontsize=8)
                im_frame.append(im1)

                # Heatmap
                im2 = axes[row_idx, 1].imshow(slice_heatmap, cmap=cmaps[pathology], vmin=0, vmax=1, animated=True)
                axes[row_idx, 1].set_title(f"{pathology} - Heatmap", fontsize=8)
                im_frame.append(im2)

                # Overlay
                im3a = axes[row_idx, 2].imshow(slice_img, cmap="bone", animated=True)
                im3b = axes[row_idx, 2].imshow(slice_heatmap, cmap=cmaps[pathology], vmin=0, vmax=1, alpha=slice_heatmap, animated=True)
                axes[row_idx, 2].set_title(f"{pathology} - Overlay", fontsize=8)
                im_frame.extend([im3a, im3b])

            for ax in axes.flatten():
                ax.axis("off")

            ims.append(im_frame)
            pbar.update(1)

        pbar.close()
        elapsed = time.time() - start_time
        print(f"Animation rendering complete in {elapsed:.2f} seconds.")

        ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=False, repeat_delay=1000)
        ani.save(save_path, writer="pillow", fps=10)
        plt.close(fig)


    def visualize_raw_attention_maps(self, image, text_tokens, labels, scan_name, original_scan_path):
        """
        Visualize per-layer, per-head attention as a grid. Each row is a head, each column a layer.
        Saves as a GIF cycling through slices along depth.
        """
        
        # Activate hooks, run forward pass and backpropagate from sim score
        self._register_hooks()
        with torch.enable_grad():
            sim_matrix, *_ = self.model(text_tokens, image)
            sim_matrix[self.rank, self.rank].backward()
        self._remove_hooks()

        # Access and process attention maps
        spatial_attention_weights = self.saved_outputs.get("spatial_attention_weights")         # torch.Size([24, 8, 576, 576])
        temporal_attention_weights = self.saved_outputs.get("temporal_attention_weights")       # torch.Size([576, 8, 24, 24])

        # Step 3: Visualize both sets
        if self.accelerator.is_main_process:
            results_dir = self._results_subdirectory("raw_attention_grids")

            self.visualize_attention_grid_gif(
                attention_weights_list=spatial_attention_weights,
                scan_name=scan_name,
                save_path=results_dir / f"{scan_name}_spatial_grid.gif",
                H=24, W=24, tokens_dim='key', mode='spatial'
            )

            self.visualize_attention_grid_gif(
                attention_weights_list=temporal_attention_weights,
                scan_name=scan_name,
                save_path=results_dir / f"{scan_name}_temporal_grid.gif",
                H=24, W=24, tokens_dim='key', mode='temporal'
            )

        return
        # Average over heads
        spatial_attn_avg = spatial_attention_weights.mean(dim=1)  # [24, 576, 576]
        temporal_attn_avg = temporal_attention_weights.mean(dim=1)  # [576, 24, 24]

        # Sum over query dimension -> attention *received* by each key token
        spatial_received = spatial_attn_avg.sum(dim=1)  # [24, 576]
        temporal_received = temporal_attn_avg.sum(dim=1)  # [576, 24]

        # Reshape: 576 -> H*W
        H, W = 24, 24
        spatial_attention_map = spatial_received.view(24, H, W)  # [D, H, W]
        temporal_attention_map = temporal_received.transpose(0, 1).view(24, H, W)

        # Normalize
        spatial_attention_map = (spatial_attention_map - spatial_attention_map.min()) / (spatial_attention_map.max() + 1e-8)
        temporal_attention_map = (temporal_attention_map - temporal_attention_map.min()) / (temporal_attention_map.max() + 1e-8)

        # Upsample and rotate attention maps
        image = image.squeeze().cpu().numpy()
        spatial_attention_map = self._upsample(spatial_attention_map, image.shape)
        temporal_attention_map = self._upsample(temporal_attention_map, image.shape)

        image = np.rot90(image, k=-1, axes=(1, 2))
        spatial_attention_map = np.rot90(spatial_attention_map, k=-1, axes=(1, 2))
        temporal_attention_map = np.rot90(temporal_attention_map, k=-1, axes=(1, 2))

        # Visualize
        if self.accelerator.is_main_process:
            results_dir = self._results_subdirectory("raw_attention_maps")
            self.visualize_overlay(image, spatial_attention_map, scan_name, "Raw Attention Map (Spatial)", results_dir / f"{scan_name}_spatial.gif", threshold=0.0, display_flags={"heatmap": True, "overlay": True})
            self.visualize_overlay(image, temporal_attention_map, scan_name, "Raw Attention Map (Temporal)", results_dir / f"{scan_name}_temporal.gif", threshold=0.0, display_flags={"heatmap": True, "overlay": True})

            np.save(results_dir / f"{scan_name}_spatial.npy", spatial_attention_map)
            np.save(results_dir / f"{scan_name}_temporal.npy", temporal_attention_map)
    
    def visualize_attention_grid_gif(self, attention_weights_list, scan_name, save_path, H=24, W=24, tokens_dim='key', mode='spatial'):
        """
        Visualize per-layer, per-head attention as a grid. Each row is a head, each column a layer.
        Saves as a GIF cycling through slices along depth (spatial) or spatial locations (temporal).

        Args:
            attention_weights_list: List[Tensor], each of shape:
                - spatial: [24, 8, 576, 576]
                - temporal: [576, 8, 24, 24]
            scan_name (str): Scan identifier for title
            save_path (Path): Where to save the GIF
            H, W (int): Spatial dimensions (default 24×24)
            tokens_dim (str): 'key' to visualize attention received, 'query' for attention sent
            mode (str): 'spatial' or 'temporal'
        """
        num_layers = len(attention_weights_list)
        num_heads = attention_weights_list[0].shape[1]
        D = 24  # depth

        attention_volumes = [[None for _ in range(num_layers)] for _ in range(num_heads)]

        for layer_idx, attn in enumerate(attention_weights_list):  # loop over layers
            for head in range(num_heads):
                if mode == 'spatial':
                    # attn: [D, H, hw, hw] → [D, hw, hw]
                    received = attn[:, head].mean(dim=1)  # sum over hw → [D, hw]
                    vol = received.view(D, H, W)  # [D, 24, 24]
                elif mode == 'temporal':
                    # attn: [HW, H, D, D] → [HW, D, D]
                    received = attn[:, head].mean(dim=1)  # sum over D → [576, 24]
                    vol = received.view(H, W, D)  # [24, 24, 24]
                    vol = torch.permute(vol, (2, 0, 1))

                vol = (vol - vol.min()) / (vol.max() + 1e-8)
                vol = vol.cpu().numpy()
                vol = np.rot90(vol, k=-1, axes=(0, 1))
                attention_volumes[head][layer_idx] = vol

        fig, axes = plt.subplots(num_heads, num_layers, figsize=(4*num_layers, 3*num_heads))
        
        if num_layers == 1:
            axes = [axes]

        ims = []

        for d in range(D):
            frame = []
            for i in range(num_heads):
                for j in range(num_layers):
                    ax = axes[i][j]
                    img = attention_volumes[i][j][d]
                    im = ax.imshow(img, cmap='inferno', vmin=0, vmax=1, animated=True)
                    if i == 0:
                        ax.set_title(f"Layer {j}", fontsize=10)
                    if j == 0:
                        ax.set_ylabel(f"Head {i}", fontsize=10)
                    frame.append(im)
                    ax.axis("off")
            ims.append(frame)

        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False, repeat_delay=1000)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ani.save(str(save_path), writer="pillow", fps=6)
        plt.close(fig)


    def attention_rollout(self, attn_weights_list, head_fusion='mean', discard_ratio=0.0, use_residual=True):
        """
        Attention rollout with optional residual connection (skip connection) handling.
        
        Args:
            attn_weights_list: List of attention matrices, each [Heads, Tokens, Tokens]
            head_fusion: 'mean' or 'max'
            discard_ratio: fraction of lowest weights to discard
            use_residual: whether to include identity skip connection at each layer

        Returns:
            rollout: [Tokens, Tokens] final attribution map
        """
        result = torch.eye(attn_weights_list[0].size(-1)).to(attn_weights_list[0].device)
        for attn in attn_weights_list:
            if head_fusion == 'mean':
                attn = attn.mean(dim=0)  # [Tokens, Tokens]
            elif head_fusion == 'max':
                attn = attn.max(dim=0)[0]
            else:
                raise ValueError(f"Unsupported head_fusion: {head_fusion}")

            if discard_ratio > 0:
                flat = attn.view(attn.shape[0], -1)
                num_discard = int(flat.shape[1] * discard_ratio)
                threshold = flat.topk(flat.shape[1] - num_discard, dim=1)[0].min(dim=1, keepdim=True)[0]
                attn = torch.where(attn >= threshold, attn, torch.zeros_like(attn))

            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

            if use_residual:
                attn = attn + torch.eye(attn.size(0)).to(attn.device)
                attn = attn / attn.sum(dim=-1, keepdim=True)

            result = attn @ result

        return result

    def attention_rollout_3d(self, attn_weights_list, D=24, H=24, W=24, head_fusion='mean',
                         discard_ratio=0.0, mode='spatial', use_residual=True):
        """
        Convert attention rollout into 3D attribution volume.

        Args:
            attn_weights_list: List of attention maps [Heads, Tokens, Tokens]
            D, H, W: Volume dimensions
            mode: 'spatial' or 'temporal'
            use_residual: Include residual connection in rollout

        Returns:
            np.ndarray: [D, H, W] attention volume
        """
        rollout = self.attention_rollout(attn_weights_list, head_fusion, discard_ratio, use_residual)

        if mode == 'spatial':
            attention_1d = rollout.sum(dim=0)  # [576]
            assert attention_1d.shape[0] == H * W, f"Expected {H*W}, got {attention_1d.shape}"
            volume_2d = attention_1d.view(H, W).cpu().numpy()
            volume = np.stack([volume_2d for _ in range(D)], axis=0)  # [D, H, W]

        elif mode == 'temporal':
            attention_1d = rollout.sum(dim=0)
            assert attention_1d.shape[0] == D, f"Expected {D}, got {attention_1d.shape}"
            volume = np.stack([np.full((H, W), v.item()) for v in attention_1d], axis=0)

        else:
            raise ValueError("Mode must be 'spatial' or 'temporal'.")

        volume = volume / (volume.max() + 1e-8)
        return volume


    def visualize_attention_rollout(self, image, text_tokens, labels, scan_name, original_scan_path):
        """
        Runs attention rollout from sim score and saves 3D attribution maps as animations.
        """
        self._register_hooks()
        with torch.enable_grad():
            sim_matrix, *_ = self.model(text_tokens, image)
            sim_matrix[self.rank, self.rank].backward()
        self._remove_hooks()
        image = image.squeeze().cpu().numpy()
        image = np.rot90(image, k=-1, axes=(1, 2))

        spatial_attention_weights = self.saved_outputs.get("spatial_attention_weights")  # [N_blocks, 8, 576, 576] or List[Tensor]
        temporal_attention_weights = self.saved_outputs.get("temporal_attention_weights")  # [576, 8, 24, 24]


        # ---- SPATIAL ROLLOUT ----
        # Assumes spatial_attention_weights is a list of blocks like [24, 8, 576, 576]
        D, H, W = 24, 24, 24
        spatial_rollouts = []

        for attn_block in spatial_attention_weights:  # each [24, 8, 576, 576]
            for d in range(attn_block.shape[0]):  # 24 depth slices
                slice_attn = attn_block[d]  # [8, 576, 576]
                rollout = self.attention_rollout(
                    [slice_attn],  # treat each depth slice as a "layer"
                    head_fusion='mean',
                    discard_ratio=0.0,
                    use_residual=True
                )
                attention_1d = rollout.sum(dim=0)  # [576]
                spatial_rollouts.append(attention_1d.view(H, W).cpu().numpy())  # [24, 24]

        # Now stack into a volume: [24, 24, 24] = [D, H, W]
        volume = np.stack(spatial_rollouts, axis=0)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        volume = self._upsample(torch.from_numpy(volume), image.shape)
        volume = np.rot90(volume, k=-1, axes=(1, 2))


        # ---- TEMPORAL ROLLOUT ----
        temporal_rollouts = []
        num_tokens = temporal_attention_weights[0].shape[0]  # assumes consistent shape
        for token_idx in range(num_tokens):
            # Extract temporal attention per layer for this token
            token_attn_layers = [layer[token_idx] for layer in temporal_attention_weights]  # List[8, 24, 24]

            # Run attention rollout over the temporal layers for this token
            rollout = self.attention_rollout(
                token_attn_layers,  # Each [8, 24, 24]
                head_fusion='mean',
                discard_ratio=0.0,
                use_residual=True
            )
            temporal_rollouts.append(rollout.sum(dim=0))  # [24] token importance over time

        temporal_vol = torch.stack(temporal_rollouts)  # [576, 24]
        temporal_vol = temporal_vol.view(H, W, D)  # [24, 24, 24]
        temporal_vol = torch.permute(temporal_vol, (2, 0, 1))

        temporal_vol = (temporal_vol - temporal_vol.min()) / (temporal_vol.max() - temporal_vol.min() + 1e-8)
        temporal_vol = self._upsample(temporal_vol, image.shape)
        temporal_vol = np.rot90(temporal_vol, k=-1, axes=(1, 2))
        
        if self.accelerator.is_main_process:
            results_dir = self._results_subdirectory("attention_rollout")
            self.visualize_overlay(image, volume, scan_name, "Attention Rollout (Spatial)", results_dir / f"{scan_name}_spatial.gif", threshold=0.0)
            self.visualize_overlay(image, temporal_vol, scan_name, "Attention Rollout (Temporal)", results_dir / f"{scan_name}_temporal.gif", threshold=0.0)

            np.save(results_dir / f"{scan_name}_spatial.npy", volume)
            np.save(results_dir / f"{scan_name}_temporal.npy", temporal_vol)

    def visualize_integrated_gradients(self, image, text_tokens, labels, scan_name, original_scan_path, steps=50):
        # Prepare baseline image
        baseline_value = 1
        baseline = torch.ones_like(image) * baseline_value
        baseline = baseline.to(self.accelerator.device)

        grads = []
        diff = image - baseline

        loss_values = []
        for alpha in torch.linspace(0, 1, steps).to(self.accelerator.device):
            interpolated = baseline + alpha * diff
            interpolated = interpolated.detach().requires_grad_()

            self.model.zero_grad()
            with torch.enable_grad():
                sim_matrix, *_ = self.model(text_tokens, interpolated)  # shape: [B, B]
                loss = sim_matrix[self.rank, self.rank]  # scalar
                loss.backward()
                loss_values.append(loss)

            grads.append(interpolated.grad.detach().clone())  # shape: [1, 1, D, H, W]

            interpolated.grad = None  # clear grad buffer
            del interpolated  # release tensor
            torch.cuda.empty_cache()

        avg_grads = torch.stack(grads).mean(dim=0)  # [1, 1, D, H, W]
        ig = (diff * avg_grads).squeeze().relu()  # [D, H, W]

        # Normalize
        ig = (ig - ig.min()) / (ig.max() + 1e-8)
        ig = ig.cpu().numpy()

        # Threshold out bottom quantile
        q_thresh = np.quantile(ig, 0.90)
        ig = np.where(ig >= q_thresh, ig, 0.0)

        # Apply contrast amplification to entire map (including zeros)
        ig = ig ** 0.05

        # Normalize again to stretch full range, including zeros
        ig = ig / (ig.max() + 1e-8)

        # threshold = np.quantile(ig_np, 0.99)
        # ig_np[ig_np < threshold] = 0

        # Rotate
        image = image.squeeze().cpu().numpy()
        image = np.rot90(image, k=-1, axes=(1, 2))
        ig = np.rot90(ig, k=-1, axes=(1, 2))

        if self.accelerator.is_main_process:
            results_dir = self._results_subdirectory("integrated_gradients")
            self.visualize_overlay(image, ig, scan_name, f"Integrated Gradients ({baseline_value})", results_dir / f"{scan_name}.gif", threshold=0.0)
            np.save(results_dir / f"{scan_name}.npy", ig)
        
        del grads, avg_grads, diff, sim_matrix, loss, baseline
        torch.cuda.empty_cache()
        gc.collect()


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


        # FF ----------------------------- FF #


        # Extract ff feature maps and gradients
        spatial_ff_features = self.saved_outputs.get("spatial_ff_features")[-1]                     # torch.Size([24, 576, 512])
        temporal_ff_features = self.saved_outputs.get("temporal_ff_features")[-1]                   # torch.Size([576, 24, 512])

        # Average over spatial tokens to get averaged weights
        spatial_ff_gradients = self.saved_outputs.get("spatial_ff_gradients")[-1].mean(dim=(0, 1))   # torch.Size([24, 576, 512]) --> torch.Size([512])
        temporal_ff_gradients = self.saved_outputs.get("temporal_ff_gradients")[-1].mean(dim=(0, 1)) # torch.Size([576, 24, 512]) --> torch.Size([512])

        # Compute ff CAMs
        spatial_ff_cam = (spatial_ff_features * spatial_ff_gradients.view(1, 1, -1)).sum(dim=-1).relu()
        temporal_ff_cam = (temporal_ff_features * temporal_ff_gradients.view(1, 1, -1)).sum(dim=-1).relu()

        # Reshape flattened ff spatial/temporal CAMs to 3D volumes
        spatial_ff_cam = spatial_ff_cam.view(24, 24, 24)
        temporal_ff_cam = temporal_ff_cam.view(24, 24, 24)
        temporal_ff_cam = torch.permute(temporal_ff_cam, (2, 0, 1))

        # Normalize ff CAMs
        spatial_ff_cam = (spatial_ff_cam - spatial_ff_cam.min()) / (spatial_ff_cam.max() + 1e-8)
        temporal_ff_cam = (temporal_ff_cam - temporal_ff_cam.min()) / (temporal_ff_cam.max() + 1e-8)


        # SELF-ATTN ----------------------------- SELF-ATTN #


        # Extract feature maps and gradients in [T, H*W, D] shape
        spatial_features = self.saved_outputs.get("spatial_features")[-1]                           # torch.Size([24, 576, 512])
        temporal_features = self.saved_outputs.get("temporal_features")[-1]                         # torch.Size([576, 24, 512])

        # Average over the embedding dimension to get weights
        spatial_gradients = self.saved_outputs.get("spatial_gradients")[-1].mean(dim=(0, 1))         # torch.Size([512])
        temporal_gradients = self.saved_outputs.get("temporal_gradients")[-1].mean(dim=(0, 1))       # torch.Size([512])

        # Compute CAMs
        spatial_cam = (spatial_features * spatial_gradients.view(1, 1, -1)).sum(dim=-1).relu()
        temporal_cam = (temporal_features * temporal_gradients.view(1, 1, -1)).sum(dim=-1).relu()

        # Reshape flattened spatial CAMs to 3D volumes
        spatial_cam = spatial_cam.view(24, 24, 24)
        temporal_cam = temporal_cam.view(24, 24, 24)
        temporal_cam = torch.permute(temporal_cam, (2, 0, 1))

        # Normalize
        spatial_cam = (spatial_cam - spatial_cam.min()) / (spatial_cam.max() + 1e-8)
        temporal_cam = (temporal_cam - temporal_cam.min()) / (temporal_cam.max() + 1e-8)

        # Combine both CAMs multiplicatively (element-wise)
        combined_cam = torch.sqrt(spatial_cam * temporal_cam + 1e-8)

        # Flattened input: [1, 13824, 512]
        vq_features = self.saved_outputs["vq_features"][-1].squeeze(0)     # [13824, 512]
        vq_gradients = self.saved_outputs["vq_gradients"][-1].squeeze(0)   # [13824, 512]

        # Step 1: Get per-channel importance weights
        weights = vq_gradients.mean(dim=0)  # [512]

        # Step 2: Apply weights and sum over channels
        vq_cam = (vq_features * weights).sum(dim=-1).relu()  # [13824]

        # Step 3: Reshape to 3D volume
        vq_cam = vq_cam.view(24, 24, 24)

        # Step 4: Normalize
        vq_cam = (vq_cam - vq_cam.min()) / (vq_cam.max() + 1e-8)

        # Upsample CAMs
        image = image.squeeze().cpu()
        spatial_cam_np = self._upsample(spatial_cam, image.shape)
        spatial_ff_cam_np = self._upsample(spatial_ff_cam, image.shape)
        temporal_cam_np = self._upsample(temporal_cam, image.shape)
        temporal_ff_cam_np = self._upsample(temporal_ff_cam, image.shape)
        combined_cam_np = self._upsample(combined_cam, image.shape)
        vq_cam_np = self._upsample(vq_cam, image.shape)

        # Rotate 90 degrees so CT table is down
        image = np.rot90(image, k=-1, axes=(1, 2))
        spatial_cam_np = np.rot90(spatial_cam_np, k=-1, axes=(1, 2))
        spatial_ff_cam_np = np.rot90(spatial_ff_cam_np, k=-1, axes=(1, 2))
        temporal_cam_np = np.rot90(temporal_cam_np, k=-1, axes=(1, 2))
        temporal_ff_cam_np = np.rot90(temporal_ff_cam_np, k=-1, axes=(1, 2))
        combined_cam_np = np.rot90(combined_cam_np, k=-1, axes=(1, 2))
        vq_cam_np = np.rot90(vq_cam_np, k=-1, axes=(1, 2))

        if self.accelerator.is_main_process:
            results_dir = self._results_subdirectory("grad_cam")
            extra_info = f"Similarity score: {sim_matrix.item()}"
            self.visualize_overlay(image, spatial_ff_cam_np, scan_name, "Grad-CAM (Spatial FeedForward)", results_dir / f"{scan_name}_spatial_ff.gif", threshold=0.0, extra_info=extra_info, display_flags={"overlay": True})
            self.visualize_overlay(image, temporal_ff_cam_np, scan_name, "Grad-CAM (Temporal FeedForward)", results_dir / f"{scan_name}_temporal_ff.gif", threshold=0.0, extra_info=extra_info, display_flags={"overlay": True})
            self.visualize_overlay(image, spatial_cam_np, scan_name, "Grad-CAM (Spatial)", results_dir / f"{scan_name}_spatial.gif", threshold=0.0, extra_info=extra_info, display_flags={"overlay": True})
            self.visualize_overlay(image, temporal_cam_np, scan_name, "Grad-CAM (Temporal)", results_dir / f"{scan_name}_temporal.gif", threshold=0.0, extra_info=extra_info, display_flags={"overlay": True})
            self.visualize_overlay(image, combined_cam_np, scan_name, "Grad-CAM (Combined)", results_dir / f"{scan_name}_combined.gif", threshold=0.0, extra_info=extra_info, display_flags={"overlay": True})
            self.visualize_overlay(image, vq_cam_np, scan_name, "Grad-CAM (VQ)", results_dir / f"{scan_name}_vq.gif", threshold=0.0, extra_info=extra_info, display_flags={"overlay": True})

            np.save(results_dir / f"{scan_name}_spatial_ff.npy", spatial_ff_cam_np)
            np.save(results_dir / f"{scan_name}_temporal_ff.npy", temporal_ff_cam_np)
            np.save(results_dir / f"{scan_name}_spatial.npy", spatial_cam_np)
            np.save(results_dir / f"{scan_name}_temporal.npy", temporal_cam_np)
            np.save(results_dir / f"{scan_name}_combined.npy", combined_cam_np)
            np.save(results_dir / f"{scan_name}_vq.npy", vq_cam_np)


    def visualize_occlusion_sensitivity(self, image, text_tokens, labels, scan_name, original_scan_path, patch_size=(20,40,40), stride=(10,20,20), use_text_embeds=False, prompt=""):
        arithmetic_embeds = np.load(self.diff_embeds_folder, allow_pickle=True)
        arithmetic_embeds = arithmetic_embeds.item()
        tensor_embeds = {k: torch.tensor(v, dtype=torch.float32).to(self.accelerator.device).unsqueeze(0) for k, v in arithmetic_embeds.items()}

        threshold = 0.0
        heatmaps = {}

        if use_text_embeds:
            positive_indices = (labels == 1).nonzero(as_tuple=True)[0]
            positive_pathologies = [PATHOLOGIES[i] for i in positive_indices.tolist()]
            for pos_pathology in positive_pathologies:
                self.maybe_print("Processing pathology:", pos_pathology)
                text_embeds = tensor_embeds[pos_pathology]
                heatmap = self._compute_occlusion(image, text_tokens, text_embeds, patch_size, stride, threshold)
                heatmaps[pos_pathology] = heatmap

        else:
            heatmap = self._compute_occlusion(image, text_tokens, None, patch_size, stride, threshold)

        if self.accelerator.is_main_process:
            extra_info = f"""
                Text: {'Pathology Embeds' if use_text_embeds else 'Report'}
                Threshold: {threshold}
                Patch size: {patch_size}
                Stride: {stride}
                Prompt: {prompt if prompt else 'No prompt'}
            """
            results_dir = self._results_subdirectory("occlusion")
            self.maybe_print("Created directory", str(results_dir))

            if use_text_embeds:
                np.save(results_dir / f"{scan_name}_{str(patch_size)}_{str(stride)}_{prompt}_heatmaps.npy", heatmaps)
            
            image = image.squeeze(0).squeeze(0)
            image = image.cpu().numpy()
            image = np.rot90(image, k=-1, axes=(1, 2))

            if use_text_embeds:
                for pathology, heatmap in heatmaps.items():
                    self.maybe_print("Visualizing the heatmap overlay for pathology:", pathology)
                    self.visualize_overlay(
                        image,
                        heatmap,
                        scan_name=f"{scan_name}_{pathology}",
                        overlay_name="Occlusion",
                        save_path=results_dir / f"{scan_name}_{pathology}_{str(patch_size)}_{str(stride)}_occlusion.gif",
                        extra_info=False,
                        display_flags={"overlay": True}
                    )
            else:
                self.maybe_print("Visualizing the heatmap overlay for prompt:", prompt)
                self.visualize_overlay(image, heatmap, scan_name, "Occlusion", results_dir / f"{scan_name}_{prompt}.gif", extra_info=False, display_flags={"overlay": True})
                np.save(results_dir / f"{scan_name}_{prompt}_heatmap.npy", heatmap)


    def visualize(self, **kwargs):
        """
        Handle visualization controls. Expects keyword arguments.
        """
        for name, val in kwargs.items():
            if not val:
                continue

            if self.world_size > 1:
                torch.distributed.barrier()

            if name in ["raw_attention_maps", "attention_rollout", "integrated_gradients", "grad_cam"]:
                dataloader = self.dist_dataloader

                if name == "raw_attention_maps":
                    visual_func = self.visualize_raw_attention_maps
                elif name == "attention_rollout":
                    visual_func = self.visualize_attention_rollout
                elif name == "integrated_gradients":
                    visual_func = self.visualize_integrated_gradients
                elif name == "grad_cam":
                    visual_func = self.visualize_grad_cam

                self.maybe_print(f"{name} visualization started.")
                start_time = time.time()

                for batch in dataloader:
                    image, texts, labels, scan_names, original_scan_paths = [
                        b.to(self.accelerator.device) if isinstance(b, torch.Tensor) else b for b in batch
                    ]

                    custom_prompt1 = "Ground-glass densities and bilateral minimal pleural effusion evaluated in favor of viral pneumonia in both lungs"
                    custom_prompt2 = "viral pneumonia"
                    text_tokens = self.tokenizer(
                        texts,
                        # custom_prompt2,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=512
                    ).to(self.accelerator.device)

                    self.model.zero_grad()
                    # black_box = torch.ones_like(image) * -1
                    visual_func(image, text_tokens, labels[0], scan_names[0], original_scan_paths[0])

            elif name == "occlusion":
                visual_func = self.visualize_occlusion_sensitivity
                self.maybe_print("Occlusion visualization started.")
                start_time = time.time()

                if self.rank == 0:
                    pbar = tqdm(total=len(self.dataset), desc="Occlusion", unit="sample", mininterval=10.0)
                else:
                    pbar = None

                for idx in range(len(self.dataset)):
                    # Load on rank 0, broadcast to all others
                    if self.rank == 0:
                        sample = self.dataset[idx]
                    else:
                        sample = None

                    if self.world_size > 1:
                        sample = self._broadcast_sample(sample)
                    else:
                        sample = [x.unsqueeze(0).to(self.accelerator.device) if isinstance(x, torch.Tensor) else [x] for x in sample]

                    image, texts, labels, scan_names, original_scan_paths = [
                        b.to(self.accelerator.device) if isinstance(b, torch.Tensor) else b for b in sample
                    ]

                    text_tokens = self.tokenizer(
                        texts,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=512
                    ).to(self.accelerator.device)

                    self.model.zero_grad()

                    # for text_prompt in SEGMENTABLE_TERMS:
                    #     self.maybe_print("Text prompt:", text_prompt)
                    #     text_tokens = self.tokenizer(
                    #         text_prompt,
                    #         return_tensors="pt",
                    #         padding="max_length",
                    #         truncation=True,
                    #         max_length=512
                    #     ).to(self.accelerator.device)
                    #     visual_func(image, text_tokens, labels[0], scan_names[0], original_scan_paths[0], prompt=text_prompt)

                    visual_func(image, text_tokens, labels[0], scan_names[0], original_scan_paths[0], use_text_embeds=False)

                    if pbar is not None:
                        pbar.update(1)

                if pbar is not None:
                    pbar.close()

            else:
                self.maybe_print(f"{name} is not a valid visualization argument.")
                return

            total_time = time.time() - start_time
            self.maybe_print(f"{name} visualization completed. Time: {str(timedelta(seconds=total_time))}")

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()