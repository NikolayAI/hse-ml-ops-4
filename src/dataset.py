from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """CSV-driven image dataset. CSV may contain a `file_name` column with a
    `train_data/<name>` prefix even for test splits, so `data_subdir` lets us
    override it with the actual subfolder on disk (e.g. `test_data`)."""

    def __init__(self, csv_file, root_dir, transform=None, data_subdir=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.data_subdir = data_subdir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fname = row["file_name"]
        if self.data_subdir is not None:
            fname = f"{self.data_subdir}/{Path(fname).name}"
        img_path = self.root_dir / fname
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
