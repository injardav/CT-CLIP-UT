import warnings
import torch
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.utils import ensure_tuple_rep
from CTCLIP import CTCLIP
from utils.CTClipInference import CTClipInference
from utils.ctvit import CTViT
from transformers import BertTokenizer, BertModel
from transformers.utils import logging
from torch import nn

warnings.simplefilter("ignore")
logging.set_verbosity_error()
torch.set_printoptions(profile="default")
torch.autograd.set_detect_anomaly(False)

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
text_encoder.resize_token_embeddings(len(tokenizer))

dim_latent = 512
dim_text = 768
vit_dim_image = 294912

# swin_dim_image = 384

# swin_encoder = SwinTransformer(
#     in_chans=1,
#     embed_dim=24,
#     window_size=ensure_tuple_rep(7, 3),
#     patch_size=ensure_tuple_rep(3, 3),
#     depths=(2, 2, 2, 2),
#     num_heads=(3, 6, 12, 24),
#     mlp_ratio=4.0,
#     qkv_bias=True,
#     drop_rate=0.0,
#     attn_drop_rate=0.0,
#     drop_path_rate=0.0,
#     norm_layer=nn.LayerNorm,
#     use_checkpoint=True,
#     spatial_dims=3,
#     downsample='merging',
#     use_v2=True
# )

vit_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 20,
    temporal_patch_size = 10,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)

clip = CTCLIP(
    text_encoder = text_encoder,
    image_encoder = vit_encoder,
    dim_text = dim_text,
    dim_image = vit_dim_image,
    dim_latent = dim_latent
)

clip.load("/mnt/ct_clip/models/CT-CLIP_v2.pt")

inference = CTClipInference(
    clip,
    valid_reports = "/mnt/ct_clip/CT-CLIP-UT/reports/valid_reports.csv",
    data_valid = "/mnt/ct_clip_data/data_volumes/dataset/valid",
    valid_labels = "/mnt/ct_clip/CT-CLIP-UT/labels/valid_labels.csv",
    valid_metadata = "/mnt/ct_clip/CT-CLIP-UT/metadata/valid_metadata.csv",
    results_folder = "/mnt/ct_clip/CT-CLIP-UT/src/results/valid",
    batch_size = 1,
    num_workers = 4,
    num_valid_samples = 1,
    zero_shot = False,
    visualize = True
)

inference.infer()
