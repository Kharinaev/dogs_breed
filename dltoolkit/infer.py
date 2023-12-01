# import sys
from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

from dltoolkit.dataset import StanfordDogsDataset
from dltoolkit.model import ResNetClassifier


def evaluate_dl(model, dl, device):
    model.eval()
    correct_num = 0
    total_num = 0
    preds_ = []
    labels_ = []
    with torch.no_grad():
        for inputs, labels in tqdm(dl, desc="Test"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.detach().cpu().argmax(1)
            correct_num += (preds == labels).sum()
            total_num += len(labels)

            preds_.append(preds)
            labels_.append(labels)

    print(f"Accuracy: {correct_num/total_num:0.6f}")

    return torch.cat(labels_), torch.cat(preds_)


def get_dataset_and_loader(**dataset_args):
    test_ds = StanfordDogsDataset(**dataset_args)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
    return test_ds, test_dl


def load_model(n_classes, model_path: Path, device):
    model = ResNetClassifier(n_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def inference():

    data_folder = Path("data")
    models_folder = Path("models")
    test_ds, test_dl = get_dataset_and_loader(
        data_folder=data_folder,
        set="test",
        transform=ToTensor(),
        dataset_path=data_folder / Path("Stanford_Dogs_256"),
        csv_path=data_folder / Path("dataset_info.csv"),
        load=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model")
    model = load_model(test_ds.n_classes, models_folder / Path("model.pth"), device)

    # Evaluation
    print("Evaluating test dataset")
    labels, preds = evaluate_dl(model, test_dl, device)

    # Save outputs
    output_df = pd.DataFrame({"true": labels, "pred": preds})
    output_df.to_csv(data_folder / Path("model_output.csv"))


if __name__ == "__main__":
    inference()
