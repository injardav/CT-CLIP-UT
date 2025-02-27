import warnings
import torch
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.utils import ensure_tuple_rep
from CTCLIP import CTCLIP
from utils.CTClipTrainer import CTClipTrainer
from transformers import BertModel
from transformers.utils import logging
from torch import nn

warnings.simplefilter("ignore")
logging.set_verbosity_error()
torch.set_printoptions(profile="default")
torch.autograd.set_detect_anomaly(False)


text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

image_encoder = SwinTransformer(
    in_chans=1,
    embed_dim=96,
    window_size=ensure_tuple_rep(3, 3),
    patch_size=(4, 4, 4),
    depths=(2, 4, 6, 4),
    num_heads=(2, 4, 8, 16),
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.05,
    attn_drop_rate=0.05,
    drop_path_rate=0.05,
    norm_layer=nn.LayerNorm,
    use_checkpoint=True,
    spatial_dims=3,
    downsample='merging',
    use_v2=True
)


clip = CTCLIP(
    text_encoder = text_encoder,
    image_encoder = image_encoder,
    dim_text = 768,
    dim_image = 1536,
    dim_latent = 512
)


trainer = CTClipTrainer(
    clip,
    train_reports= "/mnt/ct_clip_data/reports/train_reports.csv",
    valid_reports= "/mnt/ct_clip_data/reports/valid_reports.csv",
    data_train= "/mnt/ct_clip_data/train_preprocessed",
    data_valid = "/mnt/ct_clip_data/valid_preprocessed",
    valid_labels = "/mnt/ct_clip_data/labels/valid_labels.csv",
    train_metadata = "/mnt/ct_clip_data/metadata/train_metadata.csv",
    results_folder = "/mnt/ct_clip/CT-CLIP-UT/src/results",
    batch_size = 1,
    num_workers = 4,
    num_epochs = 15,
    num_save_split = 1,
    num_train_samples = 5000,
    num_valid_samples = 1000
)

trainer.train()
