import pdfplumber
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from models import Chunk
from config import settings
import chromadb


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

def store_chunks(chunks_with_embeddings: list[tuple[Chunk,list[float]]]) -> None:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("document_chunks")
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    for chunk,embedding in chunks_with_embeddings:
        ids.append(f"{chunk.source_filename}_{chunk.page_number}_{chunk.chunk_index}")
        documents.append(chunk.text)
        embeddings.append(embedding)
        metadatas.append({"source_filename": chunk.source_filename, "page_number": chunk.page_number, "chunk_index": chunk.chunk_index, "owner_id": chunk.owner_id if chunk.owner_id is not None else ""})

    collection.add(
        ids = ids,
        documents = documents,
        embeddings = embeddings,
        metadatas = metadatas
    )
    return None
