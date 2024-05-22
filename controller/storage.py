import os
import cv2
from image_sql.database import image_storage
from model.detector import get_faces
from model.recognizer import feature_extract
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import pickle

def save_lists(ids_list, embs_list, images_list, boxes_list, file_name='data.pkl'):
    with open(file_name, 'wb') as f:
        pickle.dump({
            'ids_list': ids_list,
            'embs_list': embs_list,
            'images_list': images_list,
            'boxes_list': boxes_list
        }, f)

def load_lists(file_name='data.pkl'):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        return data['ids_list'], data['embs_list'], data['images_list'], data['boxes_list']

def init_storage():
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

def load_storage():
    ids_list, embs_list, images_list, boxes_list = load_lists()
    storage = image_storage(ids_list, embs_list, images_list, boxes_list)
    return storage