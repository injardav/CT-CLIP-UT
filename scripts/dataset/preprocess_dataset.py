import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Pool
from tqdm import tqdm
import argparse
from datasets import load_dataset
from huggingface_hub import login

SCRATCH_BASE_DIR = "/mnt/ct_clip_data"
REPO_ID = 'ibrahimhamamci/CT-RATE'
HF_TOKEN = "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"

# Login to Hugging Face
login(HF_TOKEN)

def read_nii_files(directory):
    """
    Retrieve paths of all NIfTI files in the given directory.
    """
    full_path = os.path.join(SCRATCH_BASE_DIR, 'data_volumes/dataset', directory)

    nii_files = []
    for root, dirs, files in os.walk(full_path):
        for file in files:
            if file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))
    return nii_files

def read_nii_data(file_path):
    """
    Read NIfTI file data.
    """
    try:
        nii_img = nib.load(file_path)
        nii_data = nii_img.get_fdata()
        return nii_data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.
    """
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]

    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

def process_file(args):
    """
    Process a single NIfTI file.

    Args:
    args (tuple): Contains (file_path, df, split).
    """
    file_path, df, split = args
    file_name = os.path.basename(file_path)

    # Define the target folder and file path for saving the preprocessed file
    save_folder = os.path.join(SCRATCH_BASE_DIR, f"{split}_preprocessed")
    folder_path_new = os.path.join(save_folder, f"{split}_" + file_name.split("_")[1], f"{split}_" + file_name.split("_")[1] + file_name.split("_")[2])
    os.makedirs(folder_path_new, exist_ok=True)

    # Generate the final file path to check if it exists
    file_name = file_name.split(".")[0] + ".npz"
    save_path = os.path.join(folder_path_new, file_name)

    # Check if the file already exists
    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Skipping processing.")
        return

    img_data = read_nii_data(file_path)
    if img_data is None:
        print(f"Read {file_path} unsuccessful. Passing")
        return

    file_name = os.path.basename(file_path)
    row = df[df['VolumeName'] == file_name]
    if row.empty:
        print(f"No metadata found for {file_name}, skipping.")
        return

    slope = float(row["RescaleSlope"].iloc[0])
    intercept = float(row["RescaleIntercept"].iloc[0])
    xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
    z_spacing = float(row["ZSpacing"].iloc[0])

    # Define the target spacing values
    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5

    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    img_data = slope * img_data + intercept
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = ((img_data / 1000)).astype(np.float32)

    img_data = img_data.transpose(2, 0, 1)
    tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)

    resized_array = resize_array(tensor, current, target)
    resized_array = resized_array[0][0]

    np.savez(save_path, resized_array)

def main(split, metadata_df, num_workers):
    """
    Main processing function for handling both train and valid splits.

    Args:
    split (str): The split to process ('train' or 'valid').
    metadata_df (pd.DataFrame): Corresponding metadata DataFrame.
    num_workers (int): Number of worker processes.
    """
    nii_files = read_nii_files(split)  # Select the NIfTI files for the split
    with Pool(num_workers) as pool:
        pool.map(process_file, [(file, metadata_df, split) for file in nii_files])

if __name__ == "__main__":
    # Argument parser for flexibility
    parser = argparse.ArgumentParser(description="Preprocess CT-RATE NIfTI files.")
    parser.add_argument('--split', type=str, choices=['train', 'valid', 'both'], default='both',
                        help="Which data split to process: 'train', 'valid', or 'both'.")
    parser.add_argument('--workers', type=int, default=18, help="Number of workers for multiprocessing.")
    args = parser.parse_args()

    # Process 'train' split
    if args.split in ['train', 'both']:
        metadata_train_df = load_dataset(REPO_ID, "metadata")['train'].to_pandas()
        main('train', metadata_train_df, args.workers)

    # Process 'valid' split
    if args.split in ['valid', 'both']:
        metadata_valid_df = load_dataset(REPO_ID, "metadata")['validation'].to_pandas()
        main('valid', metadata_valid_df, args.workers)
