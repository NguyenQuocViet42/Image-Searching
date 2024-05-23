from pydantic import BaseModel

class SearchRequest(BaseModel):
    image: str
    table: str