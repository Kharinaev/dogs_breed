import mlflow
import numpy as np
from PIL import Image


def run_mlflow_server(model_uri, image_path=None):
    if image_path is None:
        X = np.random.randn(1, 3, 256, 256, dtype="float32")
    else:
        img = Image.open(image_path)
        X = np.array(img)[None, :]
        X = X.transpose(0, 3, 1, 2).astype("float32")

    onnx_pyfunc = mlflow.pyfunc.load_model(model_uri)
    outputs = onnx_pyfunc.predict(X)["CLASS_PROBS"]
    preds = outputs.argmax(1)
    print(preds.shape)
    print(f"Predictions: {preds}")
