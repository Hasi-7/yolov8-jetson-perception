import onnxruntime as ort
import numpy as np
import onnx
import cv2

from onnxruntime import InferenceSession
from torch import sigmoid
import torch

np.set_printoptions(suppress=True)

# Takes the input path and preprocesses it into the required format of:
    # shape(1, 3, 640, 640)
    # Color to RGB
    # Normalizes the image
def preprocess_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read the image from path: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640))
    image = image.astype(np.float32) / 255.0
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

    print("input:", image.shape, image.dtype, image.min(), image.max())

    # Running Inference Session
    sess = InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: image})

    output = outputs[0][0]
    output = output.T

    boxes       = output[:, :4]          # [8400, 4] → cx, cy, w, h
    class_logits = torch.tensor(output[:, 4:])         # [8400, 80] → per-class confidence scores

    class_scores = sigmoid(class_logits)

    conf = np.max(class_scores.numpy(), axis=1)     # [8400] → confidence score for the most likely class
    class_ids = np.argmax(class_scores.numpy(), axis=1)  # [8400] → class ID for the most likely class

    mask = conf > 0.25
    boxes = boxes[mask]
    conf = conf[mask]
    class_ids = class_ids[mask]

    xywh = []
    for (x1, y1, x2, y2) in boxes:
        xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

    indices = cv2.dnn.NMSBoxes(
        bboxes=xywh,
        scores=conf.tolist(),
        score_threshold=0.25,
        nms_threshold=0.45
    )

    if len(indices) > 0:
        indices = indices.flatten()
        boxes = boxes[indices]
        conf = conf[indices]
        class_ids = class_ids[indices]

    return outputs, boxes, conf, class_ids

def cxcywh_to_xyxy(boxes, w=640, h=640):
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    bw = boxes[:, 2]
    bh = boxes[:, 3]

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # clip to image bounds
    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    x2 = np.clip(x2, 0, w - 1)
    y2 = np.clip(y2, 0, h - 1)

    return np.stack([x1, y1, x2, y2], axis=1)

def main():
    outputs, boxes, conf, class_ids = execute_onnx_model_from_file("models/yolov8n.onnx", "data/Test_Frame.png")
    print("output shape:", outputs[0].shape, "dtype:", outputs[0].dtype, "min:", outputs[0].min(), "max:", outputs[0].max())
    print("boxes shape:", boxes.shape)
    print("conf shape:", conf.shape)
    print("class_ids shape:", class_ids.shape)

    boxes = cxcywh_to_xyxy(boxes)
    
    print("boxes after conversion to coordinates:\n", boxes)

if __name__ == "__main__":
    main()

