from fastapi import APIRouter, Depends, File, UploadFile
from model.faiss_service import FaissService
from controller import faiss_service as faiss_service_controller
from controller.start_service import get_faiss_service, images_dict, boxes_dict, table_dict, detect_model
from fastapi import FastAPI, Request
from controller.faiss_service import run_server
from model.request import SearchRequest

router = APIRouter(prefix="/faiss_service", tags=["faiss_service"])


@router.post('/search_by_file')
async def get_face_id(file: UploadFile = File(...), table_name:str = "table_test",faiss_service: FaissService = Depends(get_faiss_service)):
    return faiss_service_controller.search_by_file(file, faiss_service, images_dict, boxes_dict, detect_model, table_dict, table_name)

@router.post('/search_by_base64')
async def get_face_id(imagebase64: SearchRequest, faiss_service: FaissService = Depends(get_faiss_service)):
    return faiss_service_controller.search_by_base64(imagebase64, faiss_service, images_dict, boxes_dict, detect_model, table_dict)

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