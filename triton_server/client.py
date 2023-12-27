import hashlib
from functools import lru_cache

# from tritonclient.utils import np_to_triton_dtype
import mlflow
import numpy as np
from PIL import Image
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


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
    print("Test preprocessing")
    local = local_preproc(image_path)
    triton = call_triton_preproc(image_path)
    print(f"Shapes: Local {local.shape}, Triton {triton.shape}")
    local_hash = get_array_hash(local)
    remote_hash = get_array_hash(triton)
    assert local_hash == remote_hash
    print("Preprocessing equal\n")


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
    outputs = onnx_pyfunc.predict(X)["CLASS_PROBS"][0]
    return outputs


def test_model_outputs(image_path, model_path, gpu=False):
    triton_output = call_triton_ensemble(image_path)
    local_output = run_server(model_path, image_path)
    print(f"Classes: Local {local_output.argmax()}, Triton {triton_output.argmax()}")
    print(f"Shapes: Local {local_output.shape}, Triton {triton_output.shape}")
    if not gpu:
        triton_hash = get_array_hash(triton_output)
        local_hash = get_array_hash(local_output)
        assert triton_hash == local_hash
        print("Outputs equal")


def test_gpu_consistency(image_path, num_test=20):
    preds = []
    for _ in range(num_test):
        outputs = call_triton_ensemble(image_path)
        preds.append(outputs)

    preds = np.array(preds)
    stds = np.std(preds, axis=0)
    assert np.all(stds < 5e-4)


def main():
    image_path = "../data/Stanford_Dogs_256/n02085936-Maltese_dog/n02085936_37.jpg"
    model_path = "../mlruns/0/190888c06cca4d4a88bd3ec2b3dc8c8e/artifacts/model"

    test_preproc(image_path)
    test_model_outputs(image_path, model_path, gpu=True)
    test_gpu_consistency(image_path)


if __name__ == "__main__":
    main()
