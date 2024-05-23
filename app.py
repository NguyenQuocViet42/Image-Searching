from fastapi import APIRouter, Depends, File, UploadFile
from model.faiss_service import FaissService
from controller import faiss_service as faiss_service_controller
from start_service import get_faiss_service, images_dict, boxes_dict
from fastapi import FastAPI, Request
from controller.faiss_service import run_server

router = APIRouter(prefix="/faiss_service", tags=["faiss_service"])


@router.post('/get_face_id')
async def get_face_id(file: UploadFile = File(...), faiss_service: FaissService = Depends(get_faiss_service)):
    return faiss_service_controller.get_face_id(file, faiss_service, images_dict, boxes_dict)


if __name__ == "__main__":
    app = FastAPI()
    app.include_router(router)
    
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        response = await call_next(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['server'] = 'Embedding Service'
        return response
    
    run_server(app, 'localhost', 6969)