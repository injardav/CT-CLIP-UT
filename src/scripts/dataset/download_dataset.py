import os
import argparse
from huggingface_hub import login, hf_hub_download
from datasets import load_dataset
from tqdm import tqdm


# Constants
HF_TOKEN = ""
SCRATCH_BASE_DIR = "/mnt/ct_clip_data"
REPO_ID = 'ibrahimhamamci/CT-RATE'


# Login to Hugging Face
login(HF_TOKEN)


# Argument Parsing
parser = argparse.ArgumentParser(description="Download dataset from Hugging Face")
parser.add_argument('--split', type=str, choices=['train', 'valid', 'both'], default='both', 
                    help="Which data split to download: 'train', 'valid', or 'both'. Default is 'both'.")
parser.add_argument('--batch_size', type=int, default=100, help="Batch size for downloading data. Default is 100.")
parser.add_argument('--start_at', type=int, default=0, help="Start index for downloading records. Default is 0.")
parser.add_argument('--end_at', type=int, default=None, help="End index for downloading records. Default is len(data).")


args = parser.parse_args()


# Load datasets based on argument
split_data = {}
if args.split in ['train', 'both']:
    split_data['train'] = load_dataset(REPO_ID, "labels")['train'].to_pandas()
if args.split in ['valid', 'both']:
    split_data['valid'] = load_dataset(REPO_ID, "labels")['validation'].to_pandas()


# Define download function
def download_file_if_not_exists(subfolder, name):
    local_path = os.path.join(SCRATCH_BASE_DIR, 'data_volumes', subfolder, name)
    if not os.path.exists(local_path):
        print(f"Downloading {name} to {local_path}")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type='dataset',
            token=HF_TOKEN,
            subfolder=subfolder,
            filename=name,
            cache_dir=SCRATCH_BASE_DIR,
            local_dir=os.path.join(SCRATCH_BASE_DIR, 'data_volumes'),
            resume_download=True
        )
    else:
        print(f"File {name} already exists. Skipping download.")


# Loop over each split ('train' and 'valid') to download data
for split, data in split_data.items():
    directory_name = f'dataset/{split}/'


    # Determine the end index
    end_at = args.end_at if args.end_at else len(data)


    # Loop through the dataset in batches
    for i in tqdm(range(args.start_at, end_at, args.batch_size)):
        data_batched = data.iloc[i:i + args.batch_size]  # Get batch


        # Download each file in the batch
        for name in data_batched['VolumeName']:
            folder_parts = name.split('_')
            folder = f"{folder_parts[0]}_{folder_parts[1]}"  # e.g., train_1
            subfolder = f"{folder}_{folder_parts[2]}"  # e.g., train_1_a
            full_subfolder = os.path.join(directory_name, folder, subfolder)


            # Check if file exists, if not, download it
            download_file_if_not_exists(full_subfolder, name)


    print(f"Finished downloading {split} data.")
