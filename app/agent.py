import pdfplumber
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from models import Chunk
from config import settings

def extractor(filepath: str) -> list[tuple[int,str]]:
    with pdfplumber.open(filepath) as pdf:
        pages_list = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text == None and not text.strip()=="":
                pages_list.append((i+1,text))
        return pages_list

def chunker(pages_list: list[tuple[int,str]], filepath:str) -> list[Chunk]:
    embeddings_model = OllamaEmbeddings(
        model = settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL
    )
    counter = 0
    text_chunker = SemanticChunker(embeddings_model)
    chunks_list: list[Chunk] = []
    for page_number, page_text in pages_list:
        chunks = text_chunker.split_text(page_text)
        chunks_list.extend([Chunk(text=chunk, source_filename=filepath, page_number = page_number, chunk_index = counter+i) for i,chunk in enumerate(chunks)])
        counter += len(chunks)
    return chunks_list

def embed_chunks(chunks : list[Chunk]) -> list[tuple[Chunk,list[float]]]:
    embeddings_model = settings.EMBEDDING_MODEL
    texts = [chunk.text for chunk in chunks]
    embeddings = OllamaEmbeddings(
        model = embeddings_model,
        base_url=settings.OLLAMA_BASE_URL
    )
    embeddings_list = embeddings.embed_documents(texts)
    return_tuple : list[tuple[Chunk,list[float]]] = []
    for i in range(len(chunks)):
        return_tuple.append((chunks[i], embeddings_list[i]))
    return return_tuple


