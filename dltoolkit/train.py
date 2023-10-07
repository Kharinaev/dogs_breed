import os
import zipfile

import gdown
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from dataset import StanfordDogsDataset
from model import ResNetClassificator


def train(model, n_epochs, criterion, optimizer, dl, device):
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0.0
        correct_num = 0
        total_num = 0
        for inputs, labels in dl:

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


if __name__ == "__main__":

    # Download data
    dataset_path = "data/Stanford_Dogs_256/"
    csv_url = "https://drive.google.com/uc?id=1lWtrHY8v3QE5zg6-CzSFiY1uAKBFzadI"
    zip_url = "https://drive.google.com/uc?id=1d_0lM9PNWxH3IAkmgSXcDqBb92gec3Gd"
    csv_output = dataset_path + "dataset_info.csv"
    zip_output = dataset_path + "Stanford_Dogs_256.zip"
    gdown.download(csv_url, csv_output)
    gdown.download(zip_url, zip_output)

    with zipfile.ZipFile(zip_output, "r") as zip_ref:
        zip_ref.extractall(dataset_path)

    os.remove(zip_output)

    # Prepare dataset
    dataset_info = pd.read_csv(csv_output)
    n_classes = dataset_info.class_num.nunique()

    train_set = StanfordDogsDataset(
        dataset_path, dataset_info, set="train", transform=ToTensor()
    )
    train_dl = DataLoader(train_set, batch_size=32, shuffle=True)

    # Prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassificator(n_classes).to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train model
    train(model, 3, criterion, optimizer, train_dl, device)

    # Save model
    torch.save(model.state_dict(), "models/model.pth")
