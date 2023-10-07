from PIL import Image
from torch.utils.data import Dataset


class StanfordDogsDataset(Dataset):
    def __init__(self, dataset_path, dataset_info, set, transform):
        self.df = dataset_info[dataset_info.set == set]
        self.transform = transform
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        class_num = row["class_num"]
        image_path = row["path"]

        img = Image.open(self.dataset_path + image_path)
        if self.transform:
            img = self.transform(img)

        return img, class_num
