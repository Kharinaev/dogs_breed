import io

import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image


class TritonPythonModel:
    def initialize(self, args):
        self.image_size = (256, 256)

    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "IMAGES_SRC").as_numpy()

            image_bytes = io.BytesIO(inp.tobytes())
            image_bytes.seek(0)
            image_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
            print(image_bytes)

            # img = Image.open(image_bytes[0])
            img = Image.frombytes("RGBA", (256, 256), image_bytes, "raw")
            img = np.array(img).transpose(2, 0, 1)
            img = img / 255
            img = img[None, :]
            img = img.astype("float32")

            out = pb_utils.Tensor("IMAGES", img)
            inference_response = pb_utils.InferenceResponse(output_tensors=[out])
            responses.append(inference_response)

        return responses
