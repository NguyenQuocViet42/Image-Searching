from fastapi import APIRouter, Depends, File, UploadFile
from model.faiss_service import FaissService
from model.database import image_storage
from controller import faiss_service as faiss_service_controller
from controller.start_service import get_faiss_service, refresh_storage, add_new_image, load_storage_from_pkl, images_dict, boxes_dict, table_dict, detect_model, table_name_list, save_lists, embs_dict
from fastapi import FastAPI, Request
from controller.faiss_service import run_server
from model.request import SearchRequest, AddRequest
import numpy as np

router = APIRouter(prefix="/faiss_service", tags=["faiss_service"])

@router.post('/search_by_file')
async def search_by_file(file: UploadFile = File(...), table_name:str = "table_test",faiss_service: FaissService = Depends(get_faiss_service)):
    if table_name not in table_name_list:
        return {'error': "Table name k ton tai"}
    return faiss_service_controller.search_by_file(file, faiss_service, images_dict, boxes_dict, detect_model, table_dict, table_name)


@router.post('/search_by_base64')
async def search_by_base64(imagebase64: SearchRequest, faiss_service: FaissService = Depends(get_faiss_service)):
    if imagebase64.table not in table_name_list:
        return {'error': "Table name k ton tai"}
    return faiss_service_controller.search_by_base64(imagebase64, faiss_service, images_dict, boxes_dict, detect_model, table_dict)


@router.get('/get_list_table_name', description = 'Lấy ra tất cả tên bảng đang tồn tại')
async def get_all_table_name():
    return set(table_name_list)


@router.post('/create_new_table', description = 'Thêm một bảng mới (Sau khi thêm xong mới add được ảnh vào)')
async def create_new_table(table_name:str):
    global table_name_list
    if table_name in table_name_list:
        return {'error': "Table name da toi tai"}
    else:
        table_name_list.append(table_name)
    return {'status': "Them bang thanh cong"}


@router.delete('/delete_table', description = 'Xoá một bảng')
async def delete_table(table_name:str):
    global images_dict, boxes_dict, table_dict, embs_dict, table_name_list
    if table_name not in table_name_list:
        return {'error': "Table name khong toi tai"}
    else:
        table_name_list = [item for item in table_name_list if item != table_name]
        embs_dict, images_dict, boxes_dict, table_dict = refresh_storage(list(embs_dict.keys()), embs_dict, 
                                                                         images_dict, boxes_dict, table_dict, 
                                                                         list(table_dict.values()), table_name_list)
        
        save_lists(ids_list = list(embs_dict.keys()), embs_list = list(embs_dict.values()), 
                   images_list = list(images_dict.values()), boxes_list = list(boxes_dict.values()), 
                   list_table = list(table_dict.values()), table_name_list = table_name_list, file_name='data.pkl')
        
    return {'status': "Xoa bang thanh cong"}


@router.post('/add_new_image', description = 'Thêm 1 ảnh vào bảng')
async def create_new_table(add_request:AddRequest):
    global images_dict, boxes_dict, table_dict, embs_dict, table_name_list, faiss_service
    if add_request.table not in table_name_list:
        return {'error': "Table name khong toi tai"}
    else:
        ids_list, embs_list, images_list, boxes_list = add_new_image(   image = add_request.image, image_path = add_request.image_path, 
                                                                        ids_list = list(embs_dict.keys()), embs_list = list(embs_dict.values()), 
                                                                        images_list = list(images_dict.values()), boxes_list = list(boxes_dict.values()))
        list_table = list(table_dict.values())
        list_table.append(add_request.table)
        
        save_lists(ids_list = ids_list, embs_list = embs_list, 
                   images_list = images_list, boxes_list = boxes_list, 
                   list_table = list_table, table_name_list = table_name_list, file_name='data.pkl')
        
        storage, table_name_list = load_storage_from_pkl()
        embs_dict = storage.embs_dict
        images_dict = storage.images_dict
        boxes_dict = storage.boxes_dict
        table_dict = storage.table_dict
        ids = np.array(list(embs_dict.keys()))
        embeddings = np.array(list(embs_dict.values()))
        faiss_service = FaissService(embeddings, ids)
        
    return {'status': "Them image thanh cong"}


app = FastAPI()
app.include_router(router)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    response = await call_next(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['server'] = 'Embedding Service'
    return response

if __name__ == "__main__":
    
    run_server(app, 'localhost', 6969)