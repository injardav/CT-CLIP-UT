import torch
import math
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from pathlib import Path


class GatherWithGrad(torch.autograd.Function):
    """
    Custom autograd function to perform all_gather while preserving gradients.
    """

    @staticmethod
    def forward(ctx, tensor):
        world_size = dist.get_world_size()
        if world_size == 1:
            return tensor

        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor.contiguous())

        # Concatenate all gathered tensors along batch dimension
        gathered_tensor = torch.cat(gathered_tensors, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        Ensure gradients are correctly propagated back to the original inputs.
        """
        world_size = dist.get_world_size()
        if world_size == 1:
            return grad_output

        grad_list = list(grad_output.chunk(world_size, dim=0))
        grad_input = grad_list[dist.get_rank()]  # Extract only the gradient part belonging to this rank

        return grad_input


class CTCLIP(nn.Module):
    def __init__(self, *, text_encoder, image_encoder, dim_text, dim_image, dim_latent, temperature_init=0.07):
        """
        CTCLIP: Contrastive Text and Image Pretraining Model.

        Args:
            text_encoder (nn.Module): Pre-initialized text transformer module.
            image_encoder (nn.Module): Pre-initialized image transformer module.
            dim_text (int): Dimensionality of text embeddings.
            dim_image (int): Dimensionality of image embeddings.
            dim_latent (int): Dimensionality of the contrastive latent space.
            temperature_init (float): Initial temperature parameter for contrastive scaling.
        """
        super().__init__()

        # Encoders
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        # Projection layers
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias=False)
        self.to_image_latent = nn.Linear(dim_image, dim_latent, bias=False)

        # Temperature parameter
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature_init), dtype=torch.float))

    def load_state_dict(self, state_dict, strict=False):
        """
        Load a state dictionary into the model.
        """
        return super().load_state_dict(state_dict, strict=strict)

    def load(self, path):
        """
        Load a saved model state from a file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model state file not found at: {path}")
        
        try:
            state_dict = torch.load(str(path), map_location="cpu")  # Map to CPU to avoid device mismatch issues
            self.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded state dictionary from: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load state dictionary from {path}: {e}")

    def gather_features(self, features):
        """
        Gather features across devices while maintaining gradients.
        """
        return GatherWithGrad.apply(features)

    def forward(self, text_inputs, image_inputs):
        """
        Forward pass for CTCLIP with spatial token contrast.
        """
        # Define a helper function for printing only on the main process.
        maybe_print = print if self.accelerator.is_main_process else lambda *args, **kwargs: None

        # --- Encoding ---
        text_output = self.text_encoder(**text_inputs).last_hidden_state[:, 0, :]

        # swin encoder approach
        image_output = self.image_encoder(image_inputs)[-1]
        image_output = image_output.mean(dim=[2, 3, 4])

        # vit encoder approach
        # image_output = self.image_encoder(image_inputs)
        # image_output = image_output.mean(dim=1)
        # image_output = image_output.view(image_output.shape[0], -1)

        # --- Projection ---
        text_latents = self.to_text_latent(text_output)
        image_latents = self.to_image_latent(image_output)

        # --- Normalization ---
        text_latents = text_latents / text_latents.norm(dim=-1, keepdim=True)
        image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)

        # --- Gather across devices (if applicable) ---
        text_latents = self.gather_features(text_latents)
        image_latents = self.gather_features(image_latents)

        # --- Similarity Matrix ---
        sim_matrix = image_latents @ text_latents.t() / self.log_temperature.exp()

        return sim_matrix, image_latents, text_latents, self.log_temperature.exp()
