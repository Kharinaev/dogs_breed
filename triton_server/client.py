import io
from functools import lru_cache

import numpy as np
from PIL import Image
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    # return InferenceServerClient(url="0.0.0.0:8500")
    return InferenceServerClient(url="127.0.0.1:8500")


def call_triton_ensemble(image_path):
    triton_client = get_client()
    # with open(image_path, "rb") as image:
    #     f = image.read()
    #     bytes = bytearray(f)
    pil_img = Image.open(image_path)
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()
    image = np.array([img_byte_arr], dtype=object)
    # print(image.shape)

    input_img = InferInput(
        name="IMAGES_SRC", shape=image.shape, datatype=np_to_triton_dtype(image.dtype)
    )
    input_img.set_data_from_numpy(image, binary_data=True)

    infer_output = InferRequestedOutput("CLASS_PROBS", binary_data=True)

    query_response = triton_client.infer(
        "ensemble-onnx", [input_img], outputs=[infer_output]
    )
    class_probs = query_response.as_numpy("CLASS_PROBS")[0]
    return class_probs


# def call_triton_preproc(text: str):
#     triton_client = get_client()
#     text = np.array([text.encode("utf-8")], dtype=object)

#     input_text = InferInput(
#         name="TEXTS", shape=text.shape, datatype=np_to_triton_dtype(text.dtype)
#     )
#     input_text.set_data_from_numpy(text, binary_data=True)

#     query_response = triton_client.infer(
#         "python-tokenizer",
#         [input_text],
#         outputs=[
#             InferRequestedOutput("INPUT_IDS", binary_data=True),
#             InferRequestedOutput("ATTENTION_MASK", binary_data=True),
#         ],
#     )
#     input_ids = query_response.as_numpy("INPUT_IDS")[0]
#     attention_massk = query_response.as_numpy("ATTENTION_MASK")[0]
#     return input_ids, attention_massk


def main():
    # _input_ids, _attention_mask = call_triton_tokenizer(texts[0])
    # assert (input_ids == _input_ids).all() and (attention_mask == _attention_mask).all()

    # embeddings = torch.tensor(
    #     [call_triton_embedder_ensembele(row).tolist() for row in texts]
    # )
    # distances = torch.cdist(
    #     x1=embeddings,
    #     x2=embeddings,
    #     p=2,
    # )
    # print(distances)
    image_path = "../data/Stanford_Dogs_256/n02085936-Maltese_dog/n02085936_37.jpg"
    local_output = np.array(
        [
            1.0620191,
            -0.7587482,
            -1.6564605,
            11.401388,
            3.2964902,
            8.236326,
            -28.507414,
            3.844439,
            -23.31831,
            -12.119051,
            -7.735053,
            -2.9638703,
            25.525518,
            0.4943261,
            -9.784065,
            7.268266,
            -8.742714,
            14.933839,
            -11.886274,
            4.6339173,
            -22.943558,
            -6.0866456,
            7.4265356,
            19.42977,
            48.221542,
            0.23032856,
            9.026609,
            -0.44492602 - 12.280064,
            17.129585,
            -15.022027,
            -16.700272,
            -9.24439,
            5.7181115,
            5.2458787,
            -20.931839,
            20.451225,
            8.196779,
            13.632483,
            35.325592,
            16.890198,
            3.460497,
            -12.492832,
            6.721471,
            -7.665103,
            -5.6233363 - 33.008884,
            -2.22586,
            29.850943,
            6.8947067,
            1.4544103,
            9.806751,
            -16.165133,
            -2.354072,
            -3.7289567,
            18.51368,
            -11.124953,
            -4.79417,
            2.7173142,
            4.570362,
            -3.0870097 - 19.750126,
            -26.657274,
            16.696358,
        ]
    )

    triton_output = np.array(call_triton_ensemble(image_path))
    print(f"All close: {np.allclose(local_output, triton_output)}")
    print(f"Local: {local_output.argmax()}, Triton: {triton_output.argmax()}")


if __name__ == "__main__":
    main()
