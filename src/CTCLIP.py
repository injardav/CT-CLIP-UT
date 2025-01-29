import torch
from torch import nn
from pathlib import Path


def log(t, eps=1e-20):
    """
    Compute log(t + eps) for numerical stability.
    """
    return torch.log(t + eps)


def l2norm(t):
    """
    Normalize tensor along the last dimension using L2 norm.
    """
    return nn.functional.normalize(t, dim=-1)


def matrix_diag(t):
    """
    Extract diagonal elements of the last two dimensions.
    """
    return torch.diagonal(t, dim1=-2, dim2=-1)


class CTCLIP(nn.Module):
    def __init__(self, *, text_encoder, image_encoder, dim_text, dim_image, dim_latent, temperature_init=1.0):
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
        self.temperature = nn.Parameter(torch.tensor(temperature_init))


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


    def forward(self, text_inputs, image_inputs, return_loss=False):
        """
        Forward pass for CTCLIP.
        """
        # Encode text and images
        text_encodings = self.text_encoder(**text_inputs).last_hidden_state
        image_encodings = self.image_encoder(image_inputs)[-1]

        # Flatten and rearrange images
        image_encodings = image_encodings.flatten(2)        # Flatten spatial dimensions (depth, height, width -> depth*height*width)
        image_encodings = image_encodings.permute(0, 2, 1)  # Rearrange to (batch_size, depth*height*width, 768)

        # Project into latents
        text_latents = self.to_text_latent(text_encodings[:,0,:])
        image_latents = self.to_image_latent(image_encodings)

        # Normalize latents
        text_latents, image_latents = map(l2norm, (text_latents, image_latents))

        # Temperature
        temp = self.temperature.exp()

        # Return loss if required
        if return_loss:
            # Compute logits for text-to-image and image-to-text
            text_to_image = torch.einsum('b d, b i d -> b i', text_latents, image_latents) * temp
            image_to_text = torch.einsum('b i d, b d -> b i', image_latents, text_latents) * temp

            # Exponentiate
            text_to_image_exp, image_to_text_exp = map(torch.exp, (text_to_image, image_to_text))

            # Numerators: Positive pairs (diagonal)
            text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp))

            # Denominators: Sum over all pairs
            text_to_image_denom = text_to_image_exp.sum(dim=-1)
            image_to_text_denom = image_to_text_exp.sum(dim=-1)

            # Compute losses
            text_to_image_loss = (-log(text_to_image_pos) + log(text_to_image_denom)).mean()
            image_to_text_loss = (-log(image_to_text_pos) + log(image_to_text_denom)).mean()

            # Combine losses
            loss = (text_to_image_loss + image_to_text_loss) / 2
            return loss

        # Compute pairwise similarity (spatial scores) between text and image latents
        spatial_scores = torch.matmul(text_latents, image_latents.transpose(-1, -2)) * temp

        # Compute the global result as the mean similarity across all image tokens
        global_result = spatial_scores.mean(dim=1)

        # Return global result and spatial scores
        return global_result, spatial_scores
