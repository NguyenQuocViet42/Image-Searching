import sys
sys.path.append("controller")
from model.faiss_service import FaissService
from model.detector import YOLOv5Face, get_faces
from model.database import image_storage
from model.recognizer import feature_extract
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import pickle
import os

def save_lists(ids_list, embs_list, images_list, boxes_list, list_table, table_name_list, file_name='data.pkl'):
    with open(file_name, 'wb') as f:
        pickle.dump({
            'ids_list': ids_list,
            'embs_list': embs_list,
            'images_list': images_list,
            'boxes_list': boxes_list,
            'list_table': list_table,
            'table_name_list': table_name_list
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

def refresh_storage(ids_list, embs_dict, images_dict, boxes_dict, table_dict, table_list, table_name_list):
    # Tìm tất cả các index của các phần tử trong table_list không thuộc table_name_list
    indexes_to_remove = [index for index, table in enumerate(table_list) if table not in table_name_list]
    
    # Lấy các phần tử có index tương ứng trong ids_list
    ids_to_remove = [ids_list[index] for index in indexes_to_remove]
    
    # Xóa các phần tử có key tương ứng trong các dict
    for key in ids_to_remove:
        if key in embs_dict:
            del embs_dict[key]
        if key in images_dict:
            del images_dict[key]
        if key in boxes_dict:
            del boxes_dict[key]
        if key in table_dict:
            del table_dict[key]
    
    return embs_dict, images_dict, boxes_dict, table_dict

def load_storage_from_pkl(file_name = 'data.pkl'):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    ids_list, embs_list, images_list, boxes_list, table_list, table_name_list = data['ids_list'], data['embs_list'], data['images_list'], data['boxes_list'], data['list_table'], data['table_name_list']
    storage = image_storage(ids_list, embs_list, images_list, boxes_list, table_list)

    return storage, list(table_name_list)

def add_new_image(image, image_path, ids_list, embs_list, images_list, boxes_list):
    index = 0
    image = base64.b64decode(image)
    image = BytesIO(image)
    image = Image.open(image)
    image = np.array(image).astype(np.float32)
    image = np.array(image).astype(np.float32)
    faces, boxs = get_faces(image, detect_model)

    for i in range(len(faces)):
        emb = feature_extract(faces[i])[0][0]
        while index in ids_list:
            index += 1
        ids_list.append(index)
        embs_list.append(emb)
        images_list.append(image_path)
        boxes_list.append(boxs[i])
        index += 1
    
    return ids_list, embs_list, images_list, boxes_list

detect_model_path = 'check_point/detect.onnx'

storage, table_name_list = load_storage_from_pkl()
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