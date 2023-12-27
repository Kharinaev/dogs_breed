import fire
import mlflow
import numpy as np
from PIL import Image


def run_server(model_uri, image_path=None):
    if image_path is None:
        X = np.random.randn(1, 3, 256, 256, dtype="float32")
    else:
        img = Image.open(image_path)
        img = img.resize((256, 256))
        img = np.array(img).transpose(2, 0, 1)
        img = img / 255
        img = img[None, :]
        X = img.astype("float32")

    onnx_pyfunc = mlflow.pyfunc.load_model(model_uri)
    outputs = onnx_pyfunc.predict(X)["CLASS_PROBS"]
    preds = outputs.argmax(1)
    print(outputs)
    print(f"Predictions: {preds}")


if __name__ == "__main__":
    fire.Fire()
