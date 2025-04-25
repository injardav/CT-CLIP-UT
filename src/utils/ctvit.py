import torch
from torch import nn
from pathlib import Path
from einops import rearrange, pack, unpack
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
        model_type="ctclip"
    ):
        super().__init__()

        self.model_type = model_type
        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.patch_height = image_size // patch_size
        self.patch_width = image_size // patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)

        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(channels * patch_size**2),
            nn.Linear(channels * patch_size**2, dim),
            nn.LayerNorm(dim)
        )
        
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
        self.vq = VectorQuantize(dim=dim, codebook_size=codebook_size, use_cosine_sim=True, freeze_codebook=True if not self.training else False)
    
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
        tokens, spatial_attention_weights, spatial_cross_attention_weights = self.enc_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)
        tokens = rearrange(tokens, "(b t) (h w) d -> b t h w d", b=batch_size, h=self.patch_height, w=self.patch_width)

        # Temporal encoding
        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")
        tokens, temporal_attention_weights, temporal_cross_attention_weights = self.enc_temporal_transformer(tokens, video_shape=video_shape)
        tokens = rearrange(tokens, "(b h w) t d -> b t h w d", b=batch_size, h=self.patch_height, w=self.patch_width)

        return tokens, (spatial_attention_weights, spatial_cross_attention_weights), (temporal_attention_weights, temporal_cross_attention_weights)

    def forward(self, image, return_only_codebook_ids=False):
        if self.model_type == "ctgenerate":
            first_frame, rest_frames = image[:, :, :1], image[:, :, 1:]
            first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
            rest_frames_tokens = self.to_patch_emb(rest_frames)
            tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim = 1)
        else:
            tokens = self.to_patch_emb(image)
    
        tokens, spatial_attention_weights, temporal_attention_weights = self.encode(tokens)
        tokens, packed_fhw_shape = pack([tokens], "b * d")

        self.vq.train()    
        tokens, indices, _ = self.vq(tokens, freeze_codebook=not self.training)

        if return_only_codebook_ids:
            indices, = unpack(indices, packed_fhw_shape, 'b *')
            return indices

        tokens = rearrange(tokens, "b (t h w) d -> b t h w d", h=self.patch_height, w=self.patch_width)
        return tokens, spatial_attention_weights, temporal_attention_weights