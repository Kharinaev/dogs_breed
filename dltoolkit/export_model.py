import hydra
import mlflow
import onnx
import torch

from .config import Params
from .model import ResNetClassifier


def model_export_onnx(model, X, save_path, to_mlflow=False):
    torch.onnx.export(
        model,
        X,
        save_path,
        export_params=True,
        input_names=["IMAGES"],
        output_names=["CLASS_PROBS"],
        dynamic_axes={"IMAGES": {0: "BATCH_SIZE"}, "CLASS_PROBS": {0: "BATCH_SIZE"}},
    )
    if to_mlflow:
        onnx_model = onnx.load_model(save_path)

        with mlflow.start_run():
            signature = mlflow.models.infer_signature(
                X.numpy(), model(X).detach().numpy()
            )
            model_info = mlflow.onnx.log_model(onnx_model, "model", signature=signature)
            print(f"MLFLow onnx_model model_uri: {model_info.model_uri}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(cfg.model.num_classes, None)
    model.load_state_dict(torch.load(cfg.model.model_path, map_location=device))
    h, w = cfg.dataset.image_size
    dummy_input = torch.randn(1, 3, h, w)
    model_export_onnx(model, dummy_input, save_path=cfg.train.export_onnx, to_mlflow=True)


if __name__ == "__main__":
    main()
