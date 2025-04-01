import torch
from torch import nn
from utils.attention import Transformer, ContinuousPositionBias

class MaskGit(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        gradient_shrink_alpha = 0.1,
        heads=8,
        dim_head=64,
        attn_dropout=0.,
        ff_dropout=0.,
        **kwargs
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens + 1, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.continuous_pos_bias = ContinuousPositionBias(dim = dim_head, heads = heads, num_dims = 3)
        self.gradient_shrink_alpha = gradient_shrink_alpha

        self.transformer = Transformer(
            dim = dim,
            attn_num_null_kv = 2,
            has_cross_attn = True,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            **kwargs
        )

        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(
        self,
        ct_codebook_ids,         # [1, seq_len] — from CT-ViT
        context,                 # [1, text_len, dim] — from T5
        video_patch_shape,       # [pD, pH, pW] 
        text_mask=None,
        video_mask=None,
        return_embeds=False,
        **kwargs
    ):
        """
        Args:
            ct_codebook_ids: flattened CT token IDs [1, seq_len]
            context: T5 embeddings [1, text_len, dim]
            video_patch_shape: shape of 3D patch grid [D, H, W]
        """
        device = ct_codebook_ids.device
        b, n = ct_codebook_ids.shape

        # Token + position embeddings
        x = self.token_emb(ct_codebook_ids)  # [1, seq_len, dim]
        x = x + self.pos_emb(torch.arange(n, device=device))

        # Gradient scaling trick for smoother updates (might not be needed for inference)
        x = x * self.gradient_shrink_alpha + x.detach() * (1 - self.gradient_shrink_alpha)

        # Compute spatial attention bias
        attn_bias = self.continuous_pos_bias(*video_patch_shape, device=device)  # [num_heads, seq_len, seq_len]

        # Forward through Transformer
        x, _, cross_attn_weights = self.transformer(
            x,
            video_shape=(b, *video_patch_shape),
            context=context,
            attn_bias=attn_bias,
            self_attn_mask=video_mask,
            cross_attn_context_mask=text_mask,
            **kwargs
        )

        if return_embeds:
            return x, cross_attn_weights

        return self.to_logits(x), cross_attn_weights