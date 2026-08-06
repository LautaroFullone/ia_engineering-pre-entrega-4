"""
rag_system.py - Sistema de recuperación híbrido.

Combina dos formas de buscar, que fallan en cosas distintas:
  - Semántica (Pinecone): entiende SIGNIFICADO. Buena para lenguaje natural.
  - Léxica (BM25):        matchea PALABRAS EXACTAS. Buena para nombres propios,
                          términos técnicos, fechas.

Ambas se fusionan con EnsembleRetriever, que combina los dos rankings
(vía Reciprocal Rank Fusion) en una sola lista de resultados.

IMPORTANTE: BM25Retriever NO lee desde Pinecone; construye su índice en memoria
a partir de los chunks locales. Por eso RAGSystem carga los chunks con el mismo
loader que usó ingest.py, garantizando que ambos retrievers vean lo mismo.
"""

import os

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from loader import load_and_chunk

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "mardelplata-rag")
EMBEDDING_MODEL = "text-embedding-3-small"

# Cuántos resultados devuelve cada retriever (y el ensemble final).
TOP_K = 5


class RAGSystem:
    """Encapsula el recuperador híbrido semántico + léxico."""

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k

        # 1. Retriever SEMÁNTICO (Pinecone).
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        semantic_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

        # 2. Retriever LÉXICO (BM25), construido desde los chunks locales.
        chunks = load_and_chunk()
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = top_k

        # 3. ENSEMBLE: fusiona ambos. weights=[0.5, 0.5] les da igual peso.
        self.retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )

    def retrieve(self, query: str):
        """Devuelve los top_k documentos más relevantes combinando semántico + léxico."""
        resultados = self.retriever.invoke(query)
        # El EnsembleRetriever fusiona los resultados de ambos retrievers, por lo que
        # la unión puede superar top_k. Recortamos para que "@k" sea literal.
        return resultados[: self.top_k]


if __name__ == "__main__":
    # Prueba manual: correr `python src/rag_system.py` desde la raíz.
    rag = RAGSystem()
    consulta = "¿Cuándo se fundó Mar del Plata?"
    resultados = rag.retrieve(consulta)

    print(f"Consulta: {consulta}\n")
    print(f"Documentos recuperados ({len(resultados)}):")
    for i, doc in enumerate(resultados, start=1):
        fuente = doc.metadata.get("source", "desconocida")
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  {i}. [source: {fuente}] {preview}...")
