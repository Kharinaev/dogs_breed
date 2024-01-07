import hashlib
from functools import lru_cache

import hydra
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


try:
    from triton_server.client_config import Params
except ImportError:
    from .client_config import Params


def get_array_hash(array: np.ndarray) -> str:
    bytes_array = array.tobytes()
    hash_array = hashlib.sha256(bytes_array).hexdigest()
    return hash_array


@lru_cache
def get_client():
    return InferenceServerClient(url="127.0.0.1:8500")


def call_triton_preproc(image_path):
    triton_client = get_client()
    src_image = np.fromfile(image_path, dtype="uint8")

    infer_input = InferInput(name="IMAGES_SRC", shape=src_image.shape, datatype="UINT8")
    infer_input.set_data_from_numpy(src_image)

    infer_output = InferRequestedOutput("IMAGES")

    query_response = triton_client.infer(
        "image-preproc", [infer_input], outputs=[infer_output]
    )
    processed_image = query_response.as_numpy("IMAGES")
    return processed_image


def call_triton_ensemble(image_path):
    triton_client = get_client()
    src_image = np.fromfile(image_path, dtype="uint8")

    infer_input = InferInput(name="IMAGES_SRC", shape=src_image.shape, datatype="UINT8")
    infer_input.set_data_from_numpy(src_image)

    infer_output = InferRequestedOutput("CLASS_PROBS")

    query_response = triton_client.infer(
        "ensemble-onnx", [infer_input], outputs=[infer_output]
    )
    class_probs = query_response.as_numpy("CLASS_PROBS")
    return class_probs


def local_preproc(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = np.array(img).transpose(2, 0, 1)
    img = img / 255
    img = img[None, :]
    img = img.astype("float32")
    return img


def test_preproc(image_path):
    print("\tTest preprocessing")
    local = local_preproc(image_path)
    triton = call_triton_preproc(image_path)
    print(f"\t- Shapes: Local {local.shape}, Triton {triton.shape}")
    assert local.shape == triton.shape
    local_hash = get_array_hash(local)
    remote_hash = get_array_hash(triton)
    assert local_hash == remote_hash
    print("\tPreprocessing equal\n")


def run_local(model_path, image_path: np.array):
    img = local_preproc(image_path)
    ort_inputs = {"IMAGES": img}
    ort_session = ort.InferenceSession(model_path)
    class_probs = ort_session.run(None, ort_inputs)[0]
    return class_probs


def test_model_outputs(image_path, model_uri, gpu=False):
    print("\tTest model outputs")
    triton_output = call_triton_ensemble(image_path)
    local_output = run_local(model_uri, image_path)[0]
    print(f"\t- Classes: Local {local_output.argmax()}, Triton {triton_output.argmax()}")
    assert local_output.argmax() == triton_output.argmax()
    print(f"\t- Shapes: Local {local_output.shape}, Triton {triton_output.shape}")
    assert local_output.shape == triton_output.shape
    if not gpu:
        triton_hash = get_array_hash(triton_output)
        local_hash = get_array_hash(local_output)
        assert triton_hash == local_hash
        print("\tOutputs equal")


def test_gpu_consistency(image_path, num_test=20):
    preds = []
    for _ in range(num_test):
        outputs = call_triton_ensemble(image_path)
        preds.append(outputs)

    preds = np.array(preds)
    stds = np.std(preds, axis=0)
    assert np.all(stds < 5e-4)


def predict_image(image_path, class_num_dict):
    probas = call_triton_ensemble(image_path)
    pred = probas.argmax()
    pred_class = class_num_dict[pred].replace("_", " ")
    print(f"\nPredicted breed - {pred_class!r} for image from {image_path!r}\n")


def run_tests(test_cfg):
    image_path = test_cfg.image_path
    model_uri = test_cfg.local_model_uri_path

    print("Tests started!")
    test_preproc(image_path)
    test_model_outputs(image_path, model_uri, gpu=test_cfg.on_gpu)
    if test_cfg.on_gpu:
        test_gpu_consistency(image_path)
    print("Tests passed!")


def run_client(client_cfg: Params):
    if client_cfg.client_test.do_tests:
        run_tests(client_cfg.client_test)

    class_num_dict = pd.read_csv(client_cfg.client_run.class_num_dict_path)[
        "class"
    ].to_dict()
    predict_image(client_cfg.client_run.image_path, class_num_dict)


@hydra.main(config_path="../configs", config_name="client_config", version_base="1.3")
def main(client_cfg: Params) -> None:
    run_client(client_cfg)


if __name__ == "__main__":
    main()
