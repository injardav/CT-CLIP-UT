import torch
import math
import numpy as np
import torch.nn.functional as F
from torch import nn
from pathlib import Path


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

    def forward(self, text_inputs, image_inputs):
        """
        Forward pass for CTCLIP with spatial token contrast.
        """
        # Define a helper function for printing only on the main process.
        maybe_print = print if self.accelerator.is_main_process else lambda *args, **kwargs: None

        # --- Encoding ---
        # Get text encoding (CLS token) and image encoding (global average pooling)
        text_output = self.text_encoder(**text_inputs).last_hidden_state[:, 0, :]  # Shape: [B, text_hidden_dim]
        image_output = self.image_encoder(image_inputs)[-1].mean(dim=[2, 3, 4])    # Shape: [B, image_hidden_dim]

        maybe_print(f"text_output.grad is {text_output.grad}")
        maybe_print(f"image_output.grad is {image_output.grad}")

        # --- Projection ---
        # Project embeddings into the joint latent space
        text_latents = self.to_text_latent(text_output)     # Shape: [B, embed_dim] or [B*2, embed_dim] in validation mode
        image_latents = self.to_image_latent(image_output)  # Shape: [B, embed_dim]

        maybe_print(f"text_latents requires_grad: {text_latents.requires_grad}")
        maybe_print(f"image_latents requires_grad: {image_latents.requires_grad}")

        text_latents.retain_grad()
        image_latents.retain_grad()

        # --- Normalization ---
        text_latents = F.normalize(text_latents, p=2, dim=-1)
        image_latents = F.normalize(image_latents, p=2, dim=-1)

        maybe_print(f"1. text_latents.grad is {text_latents.grad}")
        maybe_print(f"1. image_latents.grad is {image_latents.grad}")

        # --- Gather across devices (if applicable) ---
        text_latents = self.accelerator.gather(text_latents)
        image_latents = self.accelerator.gather(image_latents)

        maybe_print(f"2. text_latents.grad is {text_latents.grad}")
        maybe_print(f"2. image_latents.grad is {image_latents.grad}")

        # --- Temperature ---
        # Ensure the temperature is positive.
        temp = self.log_temperature.exp().clamp(min=1e-6)
        maybe_print(f"Temperature (exp(log_temperature)): {temp.item():.6f}, log_temperature: {self.log_temperature.item():.6f}")
        maybe_print(f"Temperature.grad is {temp.grad}")

        # --- Similarity Matrix ---
        # Compute the similarity matrix between image and text embeddings.
        # For training mode, text_latents is assumed to have shape [B, embed_dim],
        # yielding sim_matrix of shape [B, B].
        sim_matrix = image_latents @ text_latents.t() / temp

        # --- Loss Computation ---
        batch_size = image_latents.shape[0]
        num_text_prompts = 2  # expected when two text prompts per image are provided (validation mode)

        if text_latents.shape[0] == batch_size * num_text_prompts:
            # Validation mode: each image has two associated text prompts.
            # Split text latents into two groups based on assumed interleaved order:
            # even indices: present prompts, odd indices: absent prompts.
            text_latents_present = text_latents[::2]  # Shape: [B, embed_dim]
            text_latents_absent = text_latents[1::2]  # Shape: [B, embed_dim]

            # Compute separate similarity matrices for present and absent prompts.
            sim_matrix_present = image_latents @ text_latents_present.t() / temp  # Shape: [B, B]
            sim_matrix_absent  = image_latents @ text_latents_absent.t()  / temp  # Shape: [B, B]

            # Create ground truth targets (diagonal elements correspond to positives).
            targets = torch.arange(batch_size, device=sim_matrix_present.device)

            # Compute cross-entropy losses in both directions for each prompt type.
            loss_i2t_present = F.cross_entropy(sim_matrix_present, targets)
            loss_t2i_present = F.cross_entropy(sim_matrix_present.t(), targets)
            loss_present = (loss_i2t_present + loss_t2i_present) / 2

            loss_i2t_absent  = F.cross_entropy(sim_matrix_absent, targets)
            loss_t2i_absent  = F.cross_entropy(sim_matrix_absent.t(), targets)
            loss_absent = (loss_i2t_absent + loss_t2i_absent) / 2

            # Average the losses for the two prompt types.
            loss = (loss_present + loss_absent) / 2

            # For reporting, stack the two similarity matrices.
            sim_matrix = torch.stack([sim_matrix_present, sim_matrix_absent], dim=1)  # Shape: [B, 2, B]
        else:
            # Training mode: standard single text prompt per image.
            targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
            loss_i2t = F.cross_entropy(sim_matrix, targets)
            loss_t2i = F.cross_entropy(sim_matrix.t(), targets)
            loss = (loss_i2t + loss_t2i) / 2

        # Debug prints for monitoring.
        maybe_print(f"Final loss: {loss.item()}")
        maybe_print(f"Logits mean: {sim_matrix.mean().item()}, std: {sim_matrix.std().item()}")

        maybe_print(f"Loss grad: {loss.grad}")
        maybe_print(f"Sim matrix grad: {sim_matrix.grad}")

        return loss, sim_matrix

