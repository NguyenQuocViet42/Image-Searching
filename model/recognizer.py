import torch
import onnxruntime as ort
import numpy as np
from time import time
import cv2

def create_session(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
    session = ort.InferenceSession(model_path, providers=providers)
    return session

def run_onnx_model(session, input_data):
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    result = session.run(output_names, {input_name: input_data})
    return result

model_path = 'check_point/recognize.onnx'

session = create_session(model_path)

def recognizer_warm_up():
    print("Starting Face Recognizer")
    input_data = np.random.randn(1, 3, 112, 112).astype(np.float32)
    for i in range(10):
        emb = np.array(run_onnx_model(session, input_data))
    print("Face Recognizer Ready")

def feature_extract(image):
    
    image = cv2.resize(image, (112, 112))
    image = np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0).astype(np.float32) / 255.0
    emb = np.array(run_onnx_model(session, image))
    emb = emb / np.linalg.norm(emb)
    return emb