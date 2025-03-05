import os
import glob
import torch
import pandas as pd
import numpy as np
import zipfile
from torch.utils.data import Dataset
from functools import partial


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
        self.samples = self._prepare_samples()
        
        if num_samples < len(self.samples):
            self.samples = self.samples[:num_samples]

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
        for patient_folder in glob.glob(os.path.join(self.data_folder, '*')):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                for nii_file in glob.glob(os.path.join(accession_folder, '*.npz')):
                    accession_number = os.path.basename(nii_file).replace('.npz', '.nii.gz')
                    if accession_number not in self.accession_to_text:
                        continue
                    
                    findings, impressions = self.accession_to_text[accession_number]
                    input_text = findings + impressions
                    samples.append((nii_file, input_text))

        return samples

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        nii_file, input_text = self.samples[index]
        image_tensor = self._nii_img_to_tensor(nii_file)
        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '').strip()

        return image_tensor, input_text
