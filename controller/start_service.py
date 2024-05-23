import sys
sys.path.append("controller")
from model.faiss_service import FaissService
from model.detector import YOLOv5Face, get_faces
from model.database import image_storage
from model.recognizer import feature_extract
from PIL import Image
import numpy as np
import pickle
import os

def save_lists(ids_list, embs_list, images_list, boxes_list, list_table, file_name='data.pkl'):
    with open(file_name, 'wb') as f:
        pickle.dump({
            'ids_list': ids_list,
            'embs_list': embs_list,
            'images_list': images_list,
            'boxes_list': boxes_list,
            'list_table': list_table
        }, f)

def init_storage_from_images():
    ids_list = []
    embs_list = []
    images_list = []
    boxes_list = []
    index = 0
    image_path_list = ['images/' + i for i in os.listdir('images')]
    for image_path in image_path_list:
        image = Image.open(image_path)
        image = np.array(image).astype(np.float32)
        faces, boxs = get_faces(image)

        for i in range(len(faces)):
            emb = feature_extract(faces[i])[0][0]
            ids_list.append(index)
            embs_list.append(emb)
            images_list.append(image_path)
            boxes_list.append(boxs[i])
            index += 1
    
    storage = image_storage(ids_list, embs_list, images_list, boxes_list)
    save_lists(ids_list, embs_list, images_list, boxes_list)
    
    return storage    

def load_storage_from_pkl(file_name = 'data.pkl'):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    ids_list, embs_list, images_list, boxes_list, table_list = data['ids_list'], data['embs_list'], data['images_list'], data['boxes_list'], data['list_table']
    storage = image_storage(ids_list, embs_list, images_list, boxes_list, table_list)
    
    return storage

detect_model_path = 'check_point/detect.onnx'

storage = load_storage_from_pkl()
embs_dict = storage.embs_dict
images_dict = storage.images_dict
boxes_dict = storage.boxes_dict
table_dict = storage.table_dict

ids = np.array(list(embs_dict.keys()))
embeddings = np.array(list(embs_dict.values()))

faiss_service = FaissService(embeddings, ids)
detect_model = YOLOv5Face(detect_model_path, conf_thresh=0.4, scale_roi=4)

def get_faiss_service():
    return faiss_service