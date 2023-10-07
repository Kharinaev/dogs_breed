import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from dataset import StanfordDogsDataset
from model import ResNetClassificator


def evaluate_dl(model, dl, device):
    model.eval()
    correct_num = 0
    total_num = 0
    preds_ = []
    labels_ = []
    with torch.no_grad():
        for inputs, labels in dl:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.detach().cpu().argmax(1)
            correct_num += (preds == labels).sum()
            total_num += len(labels)

            preds_.append(preds)
            labels_.append(labels)

    print(f"Accuracy: {correct_num/total_num:0.6f}")

    return torch.cat(labels_), torch.cat(preds_)


if __name__ == "__main__":

    # Prepare dataset
    print("Prepare dataset")
    dataset_path = "data/Stanford_Dogs_256/"
    dataset_info = pd.read_csv("data/dataset_info.csv")
    n_classes = dataset_info.class_num.nunique()

    test_set = StanfordDogsDataset(
        dataset_path, dataset_info, set="test", transform=ToTensor()
    )
    test_dl = DataLoader(test_set, batch_size=32, shuffle=False)

    # Load model
    print("Loading model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassificator(n_classes).to(device)
    model.load_state_dict(torch.load("models/model.pth", map_location=device))

    # Evaluation
    print("Evaluating test dataset")
    labels, preds = evaluate_dl(model, test_dl, device)

    # Save outputs
    output_df = pd.DataFrame({"true": labels, "pred": preds})
    output_df.to_csv("model_output.csv")
