import warnings
import torch
from models.ctgenerate import CTGENERATE
from utils.CTGenerateInference import CTGenerateInference
from utils.ctvit import CTViT
from utils.maskgit import MaskGit
from utils.t5 import T5Encoder

warnings.simplefilter("ignore")
torch.set_printoptions(profile="default")
torch.autograd.set_detect_anomaly(False)

ctvit = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 128,
    patch_size = 16,
    temporal_patch_size = 2,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8,
    model_type = "ctgenerate"
)

maskgit = MaskGit(
    num_tokens=8192,
    max_seq_len=10000,
    dim=512,
    dim_context=768,
    depth=6,
)

t5 = T5Encoder()

ctgenerate = CTGENERATE(
    ctvit=ctvit,
    maskgit=maskgit,
    t5=t5
)
ctgenerate.load('/mnt/ct_clip/pretrained_models/ctgenerate_filtered.pt')

inference = CTGenerateInference(
    ctgenerate,
    valid_reports = "/mnt/ct_clip/CT-CLIP-UT/reports/valid_reports.csv",
    data_valid = "/mnt/ct_clip_data/data_volumes/dataset/valid",
    valid_labels = "/mnt/ct_clip/CT-CLIP-UT/labels/valid_labels.csv",
    valid_metadata = "/mnt/ct_clip/CT-CLIP-UT/metadata/valid_metadata.csv",
    results_folder = "/mnt/ct_clip/CT-CLIP-UT/src/results/valid/ctgenerate",
    batch_size = 1,
    num_workers = 4,
    num_valid_samples = 1
)

inference.infer()
