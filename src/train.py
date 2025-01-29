from monai.networks.nets.swin_unetr import SwinTransformer
from monai.utils import ensure_tuple_rep
from CTCLIP import CTCLIP
from utils.CTClipTrainer import CTClipTrainer
from transformers import BertModel
from torch import nn


text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

image_encoder = SwinTransformer(
    in_chans=1,
    embed_dim=48,
    window_size=ensure_tuple_rep(7, 3),
    patch_size=(4,4,4),
    depths=(2, 2, 2, 2),
    num_heads=(3, 6, 12, 24),
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    norm_layer=nn.LayerNorm,
    use_checkpoint=False,
    spatial_dims=3,
    downsample='merging',
    use_v2=False,)


clip = CTCLIP(
    text_encoder = text_encoder,
    image_encoder = image_encoder,
    dim_text = 768,
    dim_image = 768,
    dim_latent = 512
)
trainer = CTClipTrainer(
    clip,
    train_reports= "/mnt/ct_clip_data/reports/train_reports.csv",
    valid_reports= "/mnt/ct_clip_data/reports/valid_reports.csv",
    data_train= "/mnt/ct_clip_data/data_volumes/dataset/train",
    data_valid = "/mnt/ct_clip_data/valid_preprocessed",
    valid_labels = "/mnt/ct_clip_data/labels/valid_labels.csv",
    train_metadata = "/mnt/ct_clip_data/metadata/train_metadata.csv",
    batch_size = 1,
    results_folder="/mnt/ct_clip_data/train_zeroshot/",
    num_train_steps = 10001,
    num_workers = 4,
)

trainer.train()
