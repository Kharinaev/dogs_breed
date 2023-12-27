import io

import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image


# import cv2


class TritonPythonModel:
    def initialize(self, args):
        self.image_size = (256, 256)

    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "IMAGES_SRC").as_numpy()

            # np_bytes = np.frombuffer(image_bytes)
            # img = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)

            img = Image.open(io.BytesIO(inp.tobytes()))
            # img = Image.frombytes("RGBA", (256, 256), image_bytes, "raw")
            # img = cv2.resize(img, self.image_size)
            img = img.resize(self.image_size)
            img = np.array(img).transpose(2, 0, 1)
            img = img / 255
            img = img[None, :]
            img = img.astype("float32")

            out = pb_utils.Tensor("IMAGES", img)
            inference_response = pb_utils.InferenceResponse(output_tensors=[out])
            responses.append(inference_response)

        return responses
