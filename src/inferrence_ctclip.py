import warnings
import torch
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.utils import ensure_tuple_rep
from models.ctclip import CTCLIP
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
--- 
clip = CTCLIP(
    text_encoder = text_encoder,
    image_encoder = vit_encoder,
    dim_text = 768,
    dim_image = 294912,
    dim_latent = 512
)

clip.load("/mnt/ct_clip/pretrained_models/ctclip_v2.pt")

inference = CTClipInference(
    clip,
    valid_reports = "/mnt/ct_clip/CT-CLIP-UT/reports/valid_reports.csv",
    data_valid = "/mnt/ct_clip_data/data_volumes/dataset/valid",
    valid_labels = "/mnt/ct_clip/CT-CLIP-UT/labels/valid_labels.csv",
    valid_metadata = "/mnt/ct_clip/CT-CLIP-UT/metadata/valid_metadata.csv",
    results_folder = "/mnt/ct_clip/CT-CLIP-UT/src/results/valid/ctclip",
    batch_size = 1,
    num_workers = 4,
    num_valid_samples = 1,
    zero_shot = False,
    visualize = True
)

inference.infer()
