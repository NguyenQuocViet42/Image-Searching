from controller.storage import init_storage, load_storage
from model.faiss_service import FaissService
from model.detector import YOLOv5Face
from time import time
import numpy as np

detect_model_path = 'check_point/detect.onnx'


storage = load_storage()
embs_dict = storage.embs_dict
images_dict = storage.images_dict
boxes_dict = storage.boxes_dict

ids = np.array(list(embs_dict.keys()))
embeddings = np.array(list(embs_dict.values()))

faiss_service = FaissService(embeddings, ids)
detect_model = YOLOv5Face(detect_model_path, conf_thresh=0.4, scale_roi=4)

def get_faiss_service():
    return faiss_service