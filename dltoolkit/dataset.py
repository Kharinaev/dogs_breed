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
        dvc_repo: Path,
        dataset_path: Path,
        csv_path: Path,
        transform=None,
        check_files: bool = False,
    ):
        if check_files:
            self.check_and_load_data(dvc_repo)

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

    def check_and_load_data(
        self,
        dvc_repo: Path,
    ):
        fs = DVCFileSystem(str(dvc_repo))
        self.fs = fs
        dvc_tracked_files = fs.find(".", detail=False, dvc_only=True)
        cnt_exists = 0
        for file in tqdm(dvc_tracked_files, desc="Checking all files"):
            if Path(file).is_file():
                cnt_exists += 1
            else:
                fs.get_file(file, file)

        print(f"Total {len(dvc_tracked_files)} files tracked by DVC")
        print(f"Already exists {cnt_exists}/{len(dvc_tracked_files)}")
