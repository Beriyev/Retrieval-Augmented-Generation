from networkx import reverse
import pdfplumber
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from models import Chunk
from config import settings
import chromadb
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from typing import Any
from threading import Lock
from models import RetrievalResult
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder


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

ensemble_retriever = None
lock = Lock()

def build_ensemble_retriever():
    global ensemble_retriever
    if ensemble_retriever is not None:
        return ensemble_retriever
    with lock:
        if ensemble_retriever is not None:
            return ensemble_retriever
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("document_chunks")
        data = collection.get(include=["documents","metadatas"])
        docs = []
        for text,meta in zip(data["documents"],data["metadatas"]): #type: ignore 
            docs.append(Document(page_content=text,metadata = meta))
        db = Chroma(
            client = client,
            collection_name = "document_chunks",
            embedding_function = OllamaEmbeddings(
                model="bge-large",
                base_url=settings.OLLAMA_BASE_URL
            )
        )
        vector_retriever = db.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k" : settings.TOP_K}
        )
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = settings.TOP_K
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.2, 0.8])
        return ensemble_retriever

def retrieve(query : str) -> list[RetrievalResult]:
    global ensemble_retriever
    if ensemble_retriever is None:
        ensemble_retriever = build_ensemble_retriever()
    results = ensemble_retriever.invoke(query)
    retrieval_list = []
    for doc in results:
        retrieval_list.append(
            RetrievalResult(
                chunk_text = doc.page_content,
                source = doc.metadata["source_filename"],
                page_number = doc.metadata["page_number"]
            )
        )
    return retrieval_list

reranker = None

def rerank(query : str, results : list[RetrievalResult]) -> list[RetrievalResult]:
    global reranker
    if reranker is None:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    pairs = []
    for result in results:
        pairs.append((query, result.chunk_text))
    scores = reranker.predict(pairs)
    ranked_results = sorted(zip(results,scores),key=lambda x : x[1],reverse = True)
    result_list = []
    for i in range(0,settings.RERANKING_TOP_K):
        result_list.append(ranked_results[i][0])
    return result_list        

llm = None
def answerer(query : str) -> str:
    global llm
    if llm is None:
        llm = ChatOllama(
            model="qwen3",
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.1
        )
    context_parts = []
    results = rerank(query, retrieve(query))
    for result in results:
        context_parts.append(f"\n\n Source:{result.source} \n Page Number:{result.page_number} \n Chunk Text:{result.chunk_text}")
    context = "".join(context_parts)
    prompt = (
        "Answer the query only based on the context below."
        "If you don't know the answer, only say that the question is out of the context that has been provided.\n\n"
        f"Context:\n{context}\n\n"
        f"Query:\n{query}\n\n"
    )
    answer = llm.invoke(prompt)
    return str(answer.content)