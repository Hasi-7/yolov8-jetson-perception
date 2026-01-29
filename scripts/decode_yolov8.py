import onnxruntime as ort
import numpy as np
import onnx

def preprocess_image(path):
    

def main():
    onnx_model = onnx.load("\\models\\yolov8n.onnx")
    onnx.checker.check_model(onnx_model)

