import hydra
import mlflow
import numpy as np
import pandas as pd
from PIL import Image


try:
    from dltoolkit.config import Params
except ImportError:
    from .config import Params


def run_server(model_uri, class_num_dict, image_path=None):
    if image_path is None:
        X = np.random.randn(1, 3, 256, 256).astype("float32")
    else:
        img = Image.open(image_path)
        img = img.resize((256, 256))
        img = np.array(img).transpose(2, 0, 1)
        img = img / 255
        img = img[None, :]
        X = img.astype("float32")

    onnx_pyfunc = mlflow.pyfunc.load_model(model_uri)
    outputs = onnx_pyfunc.predict(X)["CLASS_PROBS"]
    pred = outputs.argmax(1)[0]
    pred_class = class_num_dict[pred].replace("_", " ")
    print(f"Predicted breed: {pred_class}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    image_path = "data/Stanford_Dogs_256/n02085936-Maltese_dog/n02085936_37.jpg"

    model_uri_path = cfg.train.mlflow_model_uri_file
    with open(model_uri_path, "r") as f:
        model_uri = f.read()

    class_num_dict = pd.read_csv(cfg.infer.class_num_dict_path)["class"].to_dict()

    print(f"Using model_uri: {model_uri} for MLFlow server")
    run_server(model_uri, class_num_dict, image_path)


if __name__ == "__main__":
    main()
