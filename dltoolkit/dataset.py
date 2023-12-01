import os
import zipfile
from pathlib import Path
from typing import Optional

import gdown
import pandas as pd
from dvc.repo import Repo
from PIL import Image
from torch.utils.data import Dataset


class StanfordDogsDataset(Dataset):
    def __init__(
        self,
        set: str,
        data_folder: Path,
        csv_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        csv_url: Optional[str] = None,
        dataset_url: Optional[str] = None,
        transform=None,
        dataset_path: Optional[Path] = None,
        csv_path: Optional[Path] = None,
        load: bool = True,
    ):
        # if load:
        #     if (
        #         csv_url is None
        #         or dataset_url is None
        #         or csv_name is None
        #         or dataset_name is None
        #     ):
        #         raise ValueError(
        #             " `load` is True, but url's for loading or destination "
        #             "file names was not specified"
        #         )
        #     else:
        #         csv_path, dataset_path = self.load_dataset(
        #             data_folder, csv_name, dataset_name, csv_url, dataset_url
        #         )
        # elif dataset_path is None or csv_path is None:
        #     raise ValueError(
        #         "Dataset is not loaded because `load` is False, but `dataset_path` and/or `csv_path` was not specified"
        #     )

        # dvc pull
        self.dvc_repo = Repo("dltoolkit")
        self.dvc_repo.pull()

        self.dataset_path = Path(dataset_path)
        self.csv_path = csv_path

        df = pd.read_csv(csv_path)
        self.n_classes = df.class_num.nunique()
        self.df = df[df.set == set]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        class_num = row["class_num"]
        image_path = Path(row["path"])

        img = Image.open(self.dataset_path / image_path)
        if self.transform:
            img = self.transform(img)

        return img, class_num

    def load_dataset(
        self,
        data_folder: Path,
        csv_name: str,
        dataset_name: str,
        csv_url: str,
        dataset_url: str,
    ):
        # csv_output = "data/dataset_info.csv"
        csv_path = data_folder / Path(csv_name)
        # zip_output = "data/Stanford_Dogs_256.zip"
        dataset_path = data_folder / Path(dataset_name)
        zip_file = dataset_path.with_suffix(".zip")
        print("Downloading data")
        gdown.download(csv_url, str(csv_path))
        gdown.download(dataset_url, str(zip_file))

        print("Extracting data")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall("data")

        os.remove(zip_file)

        print("Data loaded and extracted")
        return csv_path, dataset_path
