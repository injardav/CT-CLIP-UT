# CT-CLIP Dataset Preprocessing Script

This script is used to preprocess NIfTI files from the CT-RATE dataset. It resizes the data to the target spacing and applies normalization before saving the preprocessed files in the specified scratch directory.

## Usage

### Command-Line Arguments

The script accepts the following command-line arguments:

- `--split`: Specify which data split to preprocess: `'train'`, `'valid'`, or `'both'`. Default is `'both'`.
- `--workers`: Define the number of worker processes to use for multiprocessing. Default is 18.

### Example Commands

1. **Preprocess both train and valid splits (default behavior):**

    ```bash
    python preprocess_dataset.py
    ```

2. **Preprocess only the train split using 10 workers:**

    ```bash
    python preprocess_dataset.py --split train --workers 10
    ```

3. **Preprocess the valid split with 5 workers:**

    ```bash
    python preprocess_dataset.py --split valid --workers 5
    ```

### Script Behavior

- The script reads NIfTI files from the specified split(s), resizes the images to the target spacing, and applies normalization.
- Preprocessed files are saved as `.npz` files in the corresponding directories under the scratch space.

## Notes

- Ensure the dataset has already been downloaded and is located in the scratch directory.
- Check that sufficient memory is allocated based on the number of worker processes and the size of the NIfTI files.
