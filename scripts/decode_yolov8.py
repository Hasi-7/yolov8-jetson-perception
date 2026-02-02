import onnxruntime as ort
import numpy as np
import onnx
import cv2

from onnxruntime import InferenceSession

# Takes the input path and preprocesses it into the required format of:
    # shape(1, 3, 640, 640)
    # Color to RGB
    # Normalizes the image
def preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640))
    image = image.astype((np.float32) / 255.0)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

    # Loading model using onnxruntime from file
def execute_onnx_model_from_file(model_path: str, image_path: str):
    #Loading onnx_model and checking
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    # Doing image preprocessing for for required format
    image = preprocess_image(image_path)

    # Running Inference Session
    sess = InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: image})

    return outputs

def main():
    execute_onnx_model_from_file("\\models\\yolov8n.onnx", "")
    


if __name__ == "__main__":
    main()

