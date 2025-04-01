import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import matplotlib.pyplot as plt


def read_nii_data(file_path):
    """
    Load the NIfTI file using nibabel.
    """
    try:
        nii_img = nib.load(file_path)
        nii_data = nii_img.get_fdata()
        return nii_data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

def resize_array(array, current_spacing, target_spacing):
    """
    Resize a 5D PyTorch tensor (shape: [1, 1, D, H, W]) using trilinear interpolation.
    The new shape is computed based on scaling factors derived from the spacing.

    Args:
        array (Tensor): Input tensor of shape [1, 1, D, H, W]
        current_spacing (tuple): Spacing in (z, x, y) or (D, H, W) direction
        target_spacing (tuple): Desired spacing in same order

    Returns:
        Tensor: Resized tensor of shape [1, 1, D', H', W']
    """
    original_shape = array.shape[2:]  # (D, H, W)
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False)
    return resized_array

def crop_and_pad(array, target_shape, pad_value=-1):
    """
    Crop or pad a 3D PyTorch tensor (in H, W, D order) to match the target shape.
    For each dimension:
        - if the current size is larger → center crop
        - if smaller → symmetric pad with pad_value

    Args:
        array (Tensor): Input tensor of shape [H, W, D]
        target_shape (tuple): Desired shape (H, W, D)
        pad_value (float): Value to use for padding

    Returns:
        Tensor: Tensor of shape target_shape
    """
    current_shape = array.shape
    output = array

    for i in range(3):  # for each dim: H, W, D
        dim_size = current_shape[i]
        target_size = target_shape[i]

        if dim_size > target_size:
            # Crop: center crop along dimension i
            start = (dim_size - target_size) // 2
            end = start + target_size
            if i == 0:
                output = output[start:end, :, :]
            elif i == 1:
                output = output[:, start:end, :]
            elif i == 2:
                output = output[:, :, start:end]

        elif dim_size < target_size:
            # Pad: symmetric padding along dimension i
            pad_total = target_size - dim_size
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            pad = [0, 0, 0, 0, 0, 0]  # for 3D: [D_before, D_after, W_before, W_after, H_before, H_after]
            pad[2 * (2 - i) + 0] = pad_before
            pad[2 * (2 - i) + 1] = pad_after
            output = F.pad(output, pad, mode='constant', value=pad_value)

    return output

def process_file(file_path, file_name, metadata_df, model_type):
    """
    Process a single NIfTI CT scan for the given 'model_type' pipeline using PyTorch only.

    Args:
        file_path (str): Path to the scan.
        file_name (str): Name of the scan.
        metadata_df (pd.DataFrame): DataFrame object with metadata about each scan.
        model_type (str): Name of the model that expects data.

    Returns:
        Processed CT scan as a PyTorch tensor with shape [1, 1, D, H, W].
    """
    img_data = read_nii_data(file_path)
    # img_data = np.rot90(img_data, k=-1, axes=(0, 1)).copy()

    if img_data is None:
        print(f"Read failure for {file_path}.")
        return

    row = metadata_df[metadata_df['VolumeName'] == file_name]
    if row.empty:
        print(f"No metadata found for {file_name}.")
        return

    try:
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])
    except Exception as e:
        print(f"Error processing metadata for {file_name}: {e}")
        return

    # Convert to tensor and apply HU transformation
    img_data = torch.from_numpy(img_data).float()
    img_data = slope * img_data + intercept

    # Rearrange dimensions from [H, W, D] to [D, H, W]
    img_data = img_data.permute(2, 0, 1)

    # Add batch and channel dimensions -> [1, 1, D, H, W]
    img_data = img_data.unsqueeze(0).unsqueeze(0)

    if model_type == "ctclip":
        # Resample to target spacing
        current_spacing = (z_spacing, xy_spacing, xy_spacing)
        target_spacing = (1.5, 0.75, 0.75)
        img_data = resize_array(img_data, current_spacing, target_spacing)

    # Clamp and normalize
    img_data = torch.clamp(img_data, -1000, 1000)
    img_data = img_data / 1000.0  # normalize to ~[-1, 1]

    if model_type == "ctclip":
        # Remove batch/channel dims temporarily for cropping: [D, H, W] → [H, W, D]
        img_data = img_data[0, 0].permute(1, 2, 0)

        # Crop and pad
        target_shape = (480, 480, 240)
        img_data = crop_and_pad(img_data, target_shape, pad_value=-1)

        # Return to [D, H, W], then re-add batch/channel dims
        img_data = img_data.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

    if model_type == "ctgenerate":
        img_data = F.interpolate(img_data, size=(201, 128, 128), mode='trilinear', align_corners=False)

    return img_data.squeeze(0)  # shape: [1, D, H, W]