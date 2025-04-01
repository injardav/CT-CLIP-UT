import torch
from pathlib import Path
from torch import nn

class CTGENERATE(nn.Module):
    def __init__(
        self,
        maskgit,
        ctvit,
        t5
    ):
        super().__init__()
        self.ctvit = ctvit
        self.maskgit = maskgit
        self.t5 = t5

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path, strict=True):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        self.load_state_dict(pt, strict)

    def forward(self, ct_scan, report, positive_pathologies, return_embeds=True):
        """
        Args:
            ct_scan (torch.Tensor): CT volume tensor of shape [1, 1, D, H, W]
            report (str): Single radiology report as a string
            positive_pathologies (list): List of positive pathologies for given sample

        Returns:
            feature_map and attention from maskgit
        """
        # Encode CT into discrete token IDs using CT-ViT
        ct_codebook_ids = self.ctvit(ct_scan, return_only_codebook_ids=True)  # shape: [1, D, H, W]
        video_patch_shape = ct_codebook_ids.shape[1:]  # (D, H, W) — shape of 3D patch grid before flattening
        ct_codebook_ids = ct_codebook_ids.view(1, -1)  # flatten spatial tokens → [1, seq_len]

        # Encode text using T5 encoder
        self.t5 = self.t5.to(ct_codebook_ids.device)
        text_embed = self.t5.encode(list(report))             # [1, seq_len, dim]
        text_mask = torch.any(text_embed != 0, dim=-1)        # [1, seq_len]

        # Simple binary mask for valid tokens (no masking logic needed)
        token_mask = torch.ones_like(ct_codebook_ids).bool()  # [1, seq_len]

        feature_map, attention = self.maskgit(
            ct_codebook_ids,
            context=text_embed,
            video_patch_shape=video_patch_shape,
            text_mask=text_mask,
            video_mask=token_mask,
            return_embeds=return_embeds
        )

        
        kw_attention = {}
        keyword_indices = self.t5.get_token_indices(positive_pathologies)
        for kw, indices in keyword_indices.items():
            kw_attention[kw] = attention[..., indices]
        
        return feature_map, kw_attention
