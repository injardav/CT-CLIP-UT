# Dataset Download Script

Script downloads the CT-RATE dataset from Hugging Face with flexibility to select specific data splits, set batch sizes, and define record ranges. The script will check if the files already exist in the specified directory and skip downloading those files.

## Usage

### Command-Line Arguments

The script accepts the following command-line arguments:

- `--split`: Specify which data split to download: `'train'`, `'valid'`, or `'both'` (default: `'both'`).
- `--batch_size`: Define the batch size for downloading data. Default is 100.
- `--start_at`: Specify the starting index for downloading records. Default is 0.
- `--end_at`: Specify the ending index for downloading records. Default is the total length of the dataset.

### Example Commands

1. **Download both train and validation splits (default behavior):**

    ```bash
    python download_dataset.py
    ```

2. **Download only the train split with a batch size of 50:**

    ```bash
    python download_dataset.py --split train --batch_size 50
    ```

3. **Download records from index 0 to 500 in the valid split:**

    ```bash
    python download_dataset.py --split valid --start_at 0 --end_at 500
    ```

4. **Download both splits with a batch size of 200 and start at record 1000:**

    ```bash
    python download_dataset.py --batch_size 200 --start_at 1000
    ```

### Script Behavior

- The script checks if a file already exists in the target directory. If the file exists, it will skip downloading it.
- Files are downloaded into the specified directory under `/mnt/ct_clip_data/` by default.

## Notes

- Make sure to provide your Hugging Face API token in the script to authenticate and download the dataset.
- Ensure you have enough disk space in the target directory for storing the downloaded data.
