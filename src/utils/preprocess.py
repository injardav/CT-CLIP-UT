import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F


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
    Resize a 5D tensor (shape: 1x1xD x H x W) using trilinear interpolation.
    The new shape is computed based on scaling factors derived from the spacing.
    """
    original_shape = array.shape[2:]  # (D, H, W)
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

def crop_and_pad(array, target_shape, pad_value=-1):
    """
    Crop or pad a 3D numpy array (in H, W, D order) to match the target shape.
    For each dimension, if the current size is larger than desired, we crop;
    if it’s smaller, we pad (symmetrically).
    """
    current_shape = array.shape  # (H, W, D)
    output = array.copy()
    # Process each dimension separately
    for i in range(3):
        if current_shape[i] > target_shape[i]:
            # Crop
            start = (current_shape[i] - target_shape[i]) // 2
            end = start + target_shape[i]
            if i == 0:
                output = output[start:end, :, :]
            elif i == 1:
                output = output[:, start:end, :]
            elif i == 2:
                output = output[:, :, start:end]
        elif current_shape[i] < target_shape[i]:
            # Pad
            pad_total = target_shape[i] - current_shape[i]
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            if i == 0:
                output = np.pad(output, ((pad_before, pad_after), (0,0), (0,0)), constant_values=pad_value)
            elif i == 1:
                output = np.pad(output, ((0,0), (pad_before, pad_after), (0,0)), constant_values=pad_value)
            elif i == 2:
                output = np.pad(output, ((0,0), (0,0), (pad_before, pad_after)), constant_values=pad_value)
        current_shape = output.shape
    return output

    return tensor

def process_file(file_path, file_name, metadata_df):
    """Process a single NIfTI CT scan."""
    img_data = read_nii_data(file_path)

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

    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5
    current_spacing = (z_spacing, xy_spacing, xy_spacing)
    target_spacing = (target_z_spacing, target_x_spacing, target_y_spacing)

    img_data = slope * img_data + intercept
    img_data = img_data.transpose(2, 0, 1)

    tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)

    # Resample to target spacing
    img_data = resize_array(tensor, current_spacing, target_spacing)
    img_data = img_data[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))
    
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = img_data / 1000.0

    # Crop and pad: convert (D,H,W) to (H,W,D)
    target_shape_hw_d = (480, 480, 240)
    cropped = crop_and_pad(img_data, target_shape_hw_d, pad_value=-1)
    final_array = np.transpose(cropped, (2, 0, 1))

    return final_array