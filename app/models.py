from pydantic import BaseModel, Field
from typing import Optional

class Chunk(BaseModel):
    text: str
    source_filename: str
    page_number: Optional[int] = None 
    chunk_index: int
    owner_id: Optional[str] = None

class RetrievalResult(BaseModel):
    chunk_text : str
    source : str
    page_number : int

class RAGResponse(BaseModel):
    answer: str
    query: str
    sources: list[RetrievalResult] = Field(default_factory=list)


