import os
import torch
import pandas as pd
import zipfile
from torch.utils.data import Dataset
from utils.preprocess import process_file

class InferenceDataset(Dataset):
    def __init__(self, data_folder, reports, metadata, labels, num_samples=500):
        """
        Dataset for processing CT scans and associated inference data.

        Args:
            data_folder (str): Path to the folder containing CT data.
            reports (str): Path to the report CSV file.
            metadata (str): Path to the metadata CSV file.
            labels (str): Path to the labels CSV file.
            num_samples (int, optional): Number of samples to limit the dataset. Defaults to 500.
        """
        self.data_folder = data_folder
        self.metadata_df = pd.read_csv(metadata)
        self.labels = labels
        self.observations = self._load_observations(reports)
        self.samples = self._prepare_samples()

        if num_samples < len(self.samples):
            self.samples = self.samples[:num_samples]

    def _load_observations(self, reports):
        """Load volume-to-text mapping from a CSV file."""
        df = pd.read_csv(reports)
        return {
            row['VolumeName']: (str(row['Findings_EN']) or "", str(row['Impressions_EN']) or "")
            for _, row in df.iterrows()
        }

    def _prepare_samples(self):
        """Prepare the list of samples from the data folder."""
        samples = []
        labels_df = pd.read_csv(self.labels)
        label_cols = list(labels_df.columns[1:])
        labels_df['one_hot_labels'] = list(labels_df[label_cols].values)
     
        # Traverse directory tree, find scans, fetch observations and true labels
        for root, _, files in os.walk(self.data_folder):
            for file in files:
                if file.endswith(".nii.gz"):

                    if file not in self.observations:
                        continue
                    
                    file_path = os.path.join(root, file)
                    findings, impressions = self.observations[file]
                    onehotlabels = labels_df[labels_df["VolumeName"] == file]["one_hot_labels"].values

                    if len(onehotlabels) > 0:
                        samples.append((file_path, findings + impressions, onehotlabels[0], file))

        return samples

    def __len__(self):
        return len(self.samples)

    def _preprocess_scan(self, path, name):
        """Preprocess a raw NIfTI CT scan to tensor."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        try:
            img_data = process_file(path, name, self.metadata_df)
        except zipfile.BadZipFile:
            raise RuntimeError(f"Corrupted file: {path}")
        except Exception as e:
            raise RuntimeError(f"Error loading {path}: {e}")

        return torch.tensor(img_data, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, index):
        path, observations, onehotlabels, name = self.samples[index]
        tensor = self._preprocess_scan(path, name)
        observations = observations.replace('"', '')  
        observations = observations.replace('\'', '')  
        observations = observations.replace('(', '')  
        observations = observations.replace(')', '').strip()

        return tensor, observations, onehotlabels, name
