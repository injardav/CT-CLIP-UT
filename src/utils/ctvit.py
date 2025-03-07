import torch
from torch import nn
from pathlib import Path
from einops import rearrange, pack
from einops.layers.torch import Rearrange
from vector_quantize_pytorch import VectorQuantize
from utils.attention import Attention, Transformer, ContinuousPositionBias

class CTViT(nn.Module):
    def __init__(
        self,
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=64,
        heads=8,
        channels=1,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.patch_height = image_size // patch_size
        self.patch_width = image_size // patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)
        
        self.to_patch_emb = nn.Sequential(
            Rearrange(
                "b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)",
                p1=patch_size, p2=patch_size, pt=temporal_patch_size
            ),
            nn.LayerNorm(channels * patch_size**2 * temporal_patch_size),
            nn.Linear(channels * patch_size**2 * temporal_patch_size, dim),
            nn.LayerNorm(dim),
        )

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
        )

        self.enc_spatial_transformer = Transformer(depth=spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth=temporal_depth, **transformer_kwargs)
        self.vq = VectorQuantize(dim=dim, codebook_size=codebook_size, use_cosine_sim=True)

    def load_state_dict(self, *args, **kwargs):
        """
        Load a state dictionary into the model.
        """
        return super().load_state_dict(*args, **kwargs)

    def load(self, path, strict=False):
        """
        Load a saved model state from a file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model state file not found at: {path}")
        try:
            state_dict = torch.load(str(path))
            self.load_state_dict(state_dict, strict)
            print(f"Successfully loaded state dictionary from: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load state dictionary from {path}: {e}")

    def encode(self, tokens):
        attn_bias = self.spatial_rel_pos_bias(self.patch_height, self.patch_width, device=tokens.device)
        batch_size = tokens.shape[0]
        video_shape = tokens.shape[:-1]

        # Spatial encoding
        tokens = rearrange(tokens, "b t h w d -> (b t) (h w) d")
        tokens, attention_weights = self.enc_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)
        tokens = rearrange(tokens, "(b t) (h w) d -> b t h w d", b=batch_size, h=self.patch_height, w=self.patch_width)

        # Temporal encoding
        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")
        tokens, _ = self.enc_temporal_transformer(tokens, video_shape=video_shape)
        tokens = rearrange(tokens, "(b h w) t d -> b t h w d", b=batch_size, h=self.patch_height, w=self.patch_width)

        return tokens, attention_weights

    def forward(self, image):
        tokens = self.to_patch_emb(image)
        tokens, attention_weights = self.encode(tokens)
        tokens, _ = pack([tokens], "b * d")
        tokens, _, _ = self.vq(tokens, mask=None)
        tokens = rearrange(tokens, "b (t h w) d -> b t h w d", h=self.patch_height, w=self.patch_width)
        return tokens, attention_weights