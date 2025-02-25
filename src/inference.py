from monai.networks.nets.swin_unetr import SwinTransformer
from monai.utils import ensure_tuple_rep
from CTCLIP import CTCLIP
from utils.CTClipInference import CTClipInference
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
    dim_image = 768,
    dim_text = 768,
    dim_latent = 512
)

clip.load("/mnt/ct_clip_data/train_zeroshot/CTClip.9000.pt")

inference = CTClipInference(
    clip,
    data_folder = '/mnt/ct_clip_data/valid_preprocessed',
    reports_file= "/mnt/ct_clip_data/reports/valid_reports.csv",
    labels_file = "/mnt/ct_clip_data/labels/valid_labels.csv",
    results_folder="/mnt/ct_clip_data/inference_zeroshot/",
    batch_size = 1,
    num_train_steps = 1
)

inference.infer()
