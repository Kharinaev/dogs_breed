from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

from dataset import StanfordDogsDataset
from model import ResNetClassificator
from tools import make_dirs


def train_loop(model, n_epochs, criterion, optimizer, dl, device):
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        correct_num = 0
        total_num = 0
        for inputs, labels in tqdm(dl, desc=f"Epoch {epoch+1}/{n_epochs}"):

            inputs = inputs.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            preds = outputs.detach().cpu().argmax(1)
            correct_num += (preds == labels).sum()
            total_num += len(labels)

            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(
            f"Epoch {epoch + 1}, Loss: {running_loss / len(dl)}, Accuracy: {correct_num/total_num:0.6f}"
        )

    print("Обучение завершено")


def get_dataset_and_loader(**dataset_args):
    train_ds = StanfordDogsDataset(**dataset_args)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    return train_ds, train_dl


def load_model(n_classes: int, device: torch.device):

    model = ResNetClassificator(n_classes).to(device)
    model.train()

    return model


def train_model():
    data_folder, models_folder = make_dirs(["data", "models"])
    print("Preparing dataset")
    train_ds, train_dl = get_dataset_and_loader(
        data_folder=data_folder,
        csv_name="dataset_info.csv",
        dataset_name="Stanford_Dogs_256",
        csv_url="https://drive.google.com/uc?id=12jKVnBTYlM5XtWeGgUXG-_pDwXcaRP5T",
        dataset_url="https://drive.google.com/uc?id=1d_0lM9PNWxH3IAkmgSXcDqBb92gec3Gd",
        set="train",
        transform=ToTensor(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model = load_model(train_ds.n_classes, device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Start training")
    train_loop(model, 3, criterion, optimizer, train_dl, device)
    torch.save(model.state_dict(), models_folder / Path("model.pth"))


if __name__ == "__main__":
    train_model()
