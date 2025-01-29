import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the input 3D array to match the target spacing.

    Args:
        array (torch.Tensor): Input array to be resized.
        current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
        target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
        np.ndarray: Resized array.
    """
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(current_spacing))
    ]
    new_shape = [
        int(array.shape[2 + i] * scaling_factors[i]) for i in range(len(scaling_factors))
    ]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

class TrainDataset(Dataset):
    def __init__(self, data_folder, reports, metadata, num_samples=5000):
        """
        Dataset for processing CT scans and associated text data.

        Args:
            data_folder (str): Path to the folder containing CT data.
            reports (str): Path to the report CSV file.
            metadata (str): Path to the metadata CSV file.
            num_samples (int, optional): Number of samples to limit the dataset. Defaults to 5000.
        """
        self.data_folder = data_folder
        self.metadata = metadata
        self.accession_to_text = self._load_accession_text(reports)
        self.samples = self._prepare_samples()[:num_samples]
        self.nii_to_tensor = partial(self._nii_img_to_tensor)

    def _load_accession_text(self, reports):
        """Load accession-to-text mapping from a CSV file."""
        df = pd.read_csv(reports)
        return {
            row['VolumeName']: (str(row['Findings_EN']) or "", str(row['Impressions_EN']) or "")
            for _, row in df.iterrows()
        }

    def _prepare_samples(self):
        """Prepare the list of samples from the data folder."""
        samples = []
        for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                for nii_file in glob.glob(os.path.join(accession_folder, '*.nii.gz')):
                    accession_number = os.path.basename(nii_file)
                    if accession_number not in self.accession_to_text:
                        continue
                    findings, impressions = self.accession_to_text[accession_number]
                    input_text = findings + impressions
                    samples.append((nii_file, input_text))
        return samples

    def _load_nii_metadata(self, path):
        """Load metadata from the training metadata CSV for a given file."""
        df = pd.read_csv(self.metadata)
        row = df[df['VolumeName'] == os.path.basename(path)]
        if row.empty:
            raise ValueError(f"Metadata not found for {path}")

        return {
            "slope": float(row["RescaleSlope"].iloc[0]),
            "intercept": float(row["RescaleIntercept"].iloc[0]),
            "xy_spacing": float(row["XYSpacing"].iloc[0][1:-1].split(",")[0]),
            "z_spacing": float(row["ZSpacing"].iloc[0])
        }

    def _nii_img_to_tensor(self, path):
        """Convert a NIfTI image to a tensor."""
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()
        metadata = self._load_nii_metadata(path)

        # Adjust pixel values
        img_data = metadata["slope"] * img_data + metadata["intercept"]

        # Transpose to channel-first format
        img_data = img_data.transpose(2, 0, 1)

        # Resize
        current_spacing = (metadata["z_spacing"], metadata["xy_spacing"], metadata["xy_spacing"])
        target_spacing = (1.5, 0.75, 0.75)
        img_data = resize_array(torch.tensor(img_data).unsqueeze(0).unsqueeze(0), current_spacing, target_spacing)[0][0]

        # Normalize pixel values
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = (img_data / 1000).astype(np.float32)

        # Crop and pad to target shape
        target_shape = (480, 480, 240)
        img_data = self._crop_or_pad_tensor(torch.tensor(img_data), target_shape)

        # Apply transforms
        img_data = img_data.permute(2, 0, 1).unsqueeze(0)  # Channels-first

        return img_data

    @staticmethod
    def _crop_or_pad_tensor(tensor, target_shape):
        """Crop or pad a tensor to the target shape."""
        pad_dims = [
            (max(0, (t - s) // 2), max(0, t - (s + (t - s) // 2)))
            for s, t in zip(tensor.shape, target_shape)
        ]
        tensor = F.pad(tensor, [dim for dims in reversed(pad_dims) for dim in dims], value=-1)
        slices = tuple(slice(max(0, (s - t) // 2), max(0, (s - t) // 2) + t) for s, t in zip(tensor.shape, target_shape))
        return tensor[slices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        nii_file, input_text = self.samples[index]
        image_tensor = self.nii_to_tensor(nii_file)
        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '').strip()

        return image_tensor, input_text
