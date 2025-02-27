import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from functools import partial

class InferenceDataset(Dataset):
    def __init__(self, data_folder, reports, labels, num_samples=500):
        """
        Dataset for processing CT scans and associated inference data.

        Args:
            data_folder (str): Path to the folder containing CT data.
            reports (str): Path to the CSV file containing report metadata.
            labels (str): Path to the labels CSV file.
            num_samples (int, optional): Number of samples to limit the dataset. Defaults to 500.
        """
        self.data_folder = data_folder
        self.labels = labels
        self.accession_to_text = self._load_accession_text(reports)
        self.samples = self._prepare_samples()

        if num_samples < len(self.samples):
            self.samples = self.samples[:num_samples]

    def _load_accession_text(self, reports):
        """Load accession-to-text mapping from a CSV file."""
        df = pd.read_csv(reports)
        return {
            row['VolumeName']: (row['Findings_EN'] or "", row['Impressions_EN'] or "")
            for _, row in df.iterrows()
        }

    def _prepare_samples(self):
        """Prepare the list of samples from the data folder."""
        samples = []

        # Read labels once outside the loop
        test_df = pd.read_csv(self.labels)
        test_label_cols = list(test_df.columns[1:])
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

        for patient_folder in glob.glob(os.path.join(self.data_folder, '*')):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                for nii_file in glob.glob(os.path.join(accession_folder, '*.npz')):
                    accession_number = os.path.basename(nii_file).replace('.npz', '.nii.gz')
                    if accession_number not in self.accession_to_text:
                        continue

                    findings, impressions = self.accession_to_text[accession_number]
                    text_final = findings + impressions

                    onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
                    if len(onehotlabels) > 0:
                        samples.append((nii_file, text_final, onehotlabels[0], accession_number.split('.nii.gz')[0]))
                        
        return samples

    def __len__(self):
        return len(self.samples)

    def _nii_img_to_tensor(self, path):
        """Load a preprocessed .npz file as a tensor."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        try:
            img_data = np.load(path)['arr_0']
        except zipfile.BadZipFile:
            raise RuntimeError(f"Corrupted file: {path}")
        except Exception as e:
            raise RuntimeError(f"Error loading {path}: {e}")

        return torch.tensor(img_data, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, index):
        nii_file, input_text, onehotlabels, scan_name = self.samples[index]
        image_tensor = self._nii_img_to_tensor(nii_file)
        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '').strip()

        return image_tensor, input_text, onehotlabels, scan_name
