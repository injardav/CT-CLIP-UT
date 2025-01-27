import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import tqdm

class InferenceDataset(Dataset):
    def __init__(self, data_folder, csv_file, min_slices=20, resize_dim=500, labels="labels.csv"):
        """
        Dataset for processing CT scans and associated inference data.

        Args:
            data_folder (str): Path to the folder containing CT data.
            csv_file (str): Path to the CSV file containing report metadata.
            min_slices (int, optional): Minimum number of slices for inclusion. Defaults to 20.
            resize_dim (int, optional): Dimension to resize images. Defaults to 500.
            labels (str, optional): Path to the labels CSV file. Defaults to "labels.csv".
        """
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.labels = labels
        self.accession_to_text = self._load_accession_text(csv_file)
        self.paths = []
        self.samples = self._prepare_samples()
        self.nii_to_tensor = partial(self._nii_img_to_tensor)

    def _load_accession_text(self, csv_file):
        """Load accession-to-text mapping from a CSV file."""
        df = pd.read_csv(csv_file)
        return {
            row['VolumeName']: (row['Findings_EN'] or "", row['Impressions_EN'] or "")
            for _, row in df.iterrows()
        }

    def _prepare_samples(self):
        """Prepare the list of samples from the data folder."""
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # Read labels once outside the loop
        test_df = pd.read_csv(self.labels)
        test_label_cols = list(test_df.columns[1:])
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.npz'))

                for nii_file in nii_files:
                    accession_number = os.path.basename(nii_file).replace(".npz", ".nii.gz")
                    if accession_number not in self.accession_to_text:
                        continue

                    findings, impressions = self.accession_to_text[accession_number]
                    text_final = findings + impressions

                    onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
                    if len(onehotlabels) > 0:
                        samples.append((nii_file, text_final, onehotlabels[0]))
                        self.paths.append(nii_file)
        return samples

    def __len__(self):
        return len(self.samples)

    def _nii_img_to_tensor(self, path):
        """Convert a .npz image to a tensor."""
        img_data = np.load(path)['arr_0']
        img_data = np.transpose(img_data, (1, 2, 0))
        img_data = img_data * 1000

        hu_min, hu_max = -1000, 200
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = (((img_data + 400) / 600)).astype(np.float32)

        tensor = torch.tensor(img_data)

        # Define target shape
        target_shape = (480, 480, 240)

        # Crop and pad
        tensor = self._crop_or_pad_tensor(tensor, target_shape)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # Channels-first
        return tensor

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

    def __getitem__(self, index):
        nii_file, input_text, onehotlabels = self.samples[index]
        image_tensor = self._nii_img_to_tensor(nii_file)
        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '').strip()
        name_acc = os.path.basename(os.path.dirname(nii_file))

        return image_tensor, input_text, onehotlabels, name_acc
