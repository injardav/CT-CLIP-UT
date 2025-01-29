import torch
import torch.nn as nn

class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, text_features, image_features):
        """
        text_features: (batch_size, text_len, embed_dim)
        image_features: (batch_size, num_patches, embed_dim)
        """
        attn_output, _ = self.cross_attn(text_features, image_features, image_features)  # Query = text, Key & Value = images
        attn_output = self.ln1(attn_output + text_features)  # Residual connection
        output = self.ln2(self.mlp(attn_output) + attn_output)  # Feed-forward network
        return output
