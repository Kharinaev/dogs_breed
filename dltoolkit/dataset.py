from pathlib import Path

import pandas as pd
from dvc.api import DVCFileSystem
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class StanfordDogsDataset(Dataset):
    def __init__(
        self,
        set: str,
        abs_dvc_repo: Path,
        dataset_path: Path,
        csv_path: Path,
        transform=None,
        load: bool = True,
    ):
        if load:
            self.load_data(abs_dvc_repo)

        self.dataset_path = Path(dataset_path)
        self.csv_path = csv_path

        df = pd.read_csv(csv_path)
        self.set = set
        self.n_classes = df.class_num.nunique()
        self.df = df[df.set == set]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        class_num = row["class_num"]
        image_path = Path(row["path"])

        img = Image.open(str(self.dataset_path / image_path))
        if self.transform:
            img = self.transform(img)

        return img, class_num

    def load_data(
        self,
        abs_dvc_repo: Path,
    ):
        from dvc.repo import Repo

        repo = Repo(str(abs_dvc_repo))
        repo.pull(force=True)