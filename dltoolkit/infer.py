from pathlib import Path

import hydra
import pandas as pd
import torch
from sklearn.metrics import top_k_accuracy_score
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm


try:
    from dltoolkit.config import DatasetConfig, InferConfig, Params
    from dltoolkit.dataset import StanfordDogsDataset
    from dltoolkit.model import ResNetClassifier
except ImportError:
    from config import DatasetConfig, InferConfig, Params
    from dataset import StanfordDogsDataset
    from model import ResNetClassifier


def evaluate_dl(model, dl, device, acc_topk):
    model.eval()
    correct_num = {k: 0 for k in acc_topk}
    total_num = 0
    preds_ = []
    labels_ = []
    with torch.no_grad():
        for inputs, labels in tqdm(dl, desc="Test"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.detach().cpu().argmax(1)
            # correct_num += (preds == labels).sum()
            for k in acc_topk:
                cn = top_k_accuracy_score(
                    y_true=labels,
                    y_score=outputs,
                    k=k,
                    normalize=False,
                    labels=list(range(outputs.shape[1])),
                )
                correct_num[k] += cn
            total_num += len(labels)

            preds_.append(preds)
            labels_.append(labels)

    for k in acc_topk:
        print(f"Accuracy top-{k}: {correct_num[k]/total_num:0.6f}")

    return torch.cat(labels_), torch.cat(preds_)


def load_model(n_classes, model_path: Path, device):
    model = ResNetClassifier(n_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def inference(infer_cfg: InferConfig, dataset_cfg: DatasetConfig):
    test_ds = StanfordDogsDataset(
        "test",
        abs_dvc_repo=Path.cwd(),
        dataset_path=Path(dataset_cfg.dataset_path),
        csv_path=Path(dataset_cfg.csv_path),
        transform=ToTensor(),
    )
    test_dl = DataLoader(test_ds, batch_size=dataset_cfg.batch_size, shuffle=False)

    print("Loading model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(test_ds.n_classes, infer_cfg.model_load_path, device)

    print("Evaluating test dataset")
    labels, preds = evaluate_dl(model, test_dl, device, acc_topk=infer_cfg.accuracy_topk)

    output_df = pd.DataFrame({"true": labels, "pred": preds})
    output_df.to_csv(infer_cfg.csv_output_save_path)
    print(f"Outputs saved to {infer_cfg.csv_output_save_path}")


@hydra.main(config_path="../configs", config_name="config_subset", version_base="1.3")
def main(cfg: Params) -> None:
    inference(cfg.infer, cfg.dataset)


if __name__ == "__main__":
    main()
