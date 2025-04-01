import os
import torch
import pandas as pd
import zipfile
from torch.utils.data import Dataset
from utils.preprocess import process_file

class InferenceDataset(Dataset):
    def __init__(self, data_folder, reports, metadata, labels, num_samples=500, model_type="ctclip"):
        """
        Dataset for processing CT scans and associated inference data.

        Args:
            data_folder (str): Path to the folder containing CT data.
            reports (str): Path to the report CSV file.
            metadata (str): Path to the metadata CSV file.
            labels (str): Path to the labels CSV file.
            num_samples (int, optional): Number of samples to limit the dataset. Defaults to 500.
            model_type (str, optional): Type of the model, either 'ctclip' or 'ctgenerate'. Defaults to 'ctclip'.
        """
        self.data_folder = data_folder
        self.metadata_df = pd.read_csv(metadata)
        self.labels = labels
        self.observations = self._load_observations(reports)
        self.samples = self._prepare_samples()
        self.model_type = model_type

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
                    labels = labels_df[labels_df["VolumeName"] == file]["one_hot_labels"].values

                    if len(labels) > 0:
                        samples.append((file_path, findings + impressions, labels[0], file))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, observations, labels, name = self.samples[index]
        image = process_file(path, name, self.metadata_df, self.model_type)
        name = name.replace(".nii.gz", "")
        labels = torch.tensor(labels, dtype=torch.float32)
        observations = observations.replace('"', '')  
        observations = observations.replace('\'', '')  
        observations = observations.replace('(', '')  
        observations = observations.replace(')', '').strip()

        return image, observations, labels, name, path
