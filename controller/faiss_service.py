import numpy as np
from fastapi import UploadFile
from model.faiss_service import FaissService
import os
import uvicorn
from fastapi import FastAPI
from model.recognizer import feature_extract
from tools.detect_tools import get_faces, draw_box
from PIL import Image
from io import BytesIO
import cv2

def get_face_id(file: UploadFile, faiss_service:FaissService, images_dict, boxes_dict):
    try:
        files = os.listdir('result')
        for name in files:
            os.remove('result/' + name)
        contents = file.file.read()
        image = Image.open(BytesIO(contents))
        image = np.array(image).astype(np.float32)
        face, image_list = get_faces(image)
        face = face[0]
        cv2.imwrite("result/1_search.png", face)
        
        emb = feature_extract(face)
        list_ressult_path = []

        D, I = faiss_service.search(emb[0], 20)

        for i in range(D.shape[1]):
            print(float(D[0][i]))
            if float(D[0][i]) >= 0.395:
                list_ressult_path.append(images_dict[I[0][i]])
                box =  boxes_dict[I[0][i]]
                image_drawed = draw_box(images_dict[I[0][i]], box)
                cv2.imwrite("result/{}.png".format(i), image_drawed)
        return {'path_image_list': list_ressult_path}
            
    except Exception as e:
        print(e)
        return {'path_image_list': -1}
        
        
def run_server(app:FastAPI, host, port):
    try:
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        print(e)
    finally:
        os._exit(1)