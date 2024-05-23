from pydantic import BaseModel

class SearchRequest(BaseModel):
    image: str
    table: str
    
class AddRequest(BaseModel):
    image: str
    table: str
    image_path: str