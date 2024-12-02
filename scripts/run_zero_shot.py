import torch
import argparse
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from zero_shot import CTClipInference
import accelerate

def main():
    parser = argparse.ArgumentParser(description="Run CT-CLIP inference.")
    parser.add_argument("--save_weights", action="store_true", help="Flag to save attention weights during inference.")
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder.resize_token_embeddings(len(tokenizer))


    image_encoder = CTViT(
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
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_image = 294912,
        dim_text = 768,
        dim_latent = 512,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False

    )

    clip.load("/mnt/ct_clip/models/CT-CLIP_v2.pt")

    inference = CTClipInference(
        clip,
        data_folder = '/mnt/ct_clip_data/valid_preprocessed',
        reports_file= "/mnt/ct_clip_data/reports/valid_reports.csv",
        labels = "/mnt/ct_clip_data/labels/valid_labels.csv",
        batch_size = 1,
        results_folder="/mnt/ct_clip_home/inference_zeroshot/",
        num_train_steps = 1,
        save_weights = args.save_weights
    )

    print("Running inference...")
    inference.infer()
    print("Inference completed. Results saved to: /mnt/ct_clip_data/inference_zeroshot/")

if __name__ == "__main__":
    main()
