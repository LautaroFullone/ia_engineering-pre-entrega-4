"""
ingest.py - Pipeline de ingesta a Pinecone.

Flujo:
  1. Carga y fragmenta los documentos (via loader.load_and_chunk).
  2. Verifica que el índice Serverless exista; si no, lo crea.
  3. Sube los chunks a Pinecone usando PineconeVectorStore, que internamente
     genera los embeddings (text-embedding-3-small) y hace el upsert.

Uso:
    python src/ingest.py

Requiere las variables de entorno PINECONE_API_KEY, OPENAI_API_KEY e INDEX_NAME
(cargadas desde un archivo .env).
"""

import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from loader import load_and_chunk

# Cargar variables de entorno desde .env
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "mardelplata-rag")

# text-embedding-3-small produce vectores de 1536 dimensiones y se optimiza
# para similitud coseno. Estos dos valores DEBEN coincidir en el índice.
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 1536


def ensure_index(pinecone: Pinecone, index_name: str, dimension: int):
    """Crea el índice Serverless si no existe (idempotente)."""
    existing = [idx["name"] for idx in pinecone.list_indexes()]

    if index_name not in existing:
        print(f"Creando índice '{index_name}'...")
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Esperar a que el índice esté operativo antes de usarlo.
        while not pinecone.describe_index(index_name).status["ready"]:
            time.sleep(1)

        print("Índice creado y listo.")
    else:
        print(f"El índice '{index_name}' ya existe. Se reutiliza.")


def main():
    # 1. Cargar y fragmentar los documentos.
    print("Cargando y fragmentando documentos de /data...")
    chunks = load_and_chunk()
    print(f"{len(chunks)} chunks generados.")

    # 2. Inicializar cliente Pinecone y asegurar el índice.
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pinecone, INDEX_NAME, DIMENSION)
    index = pinecone.Index(INDEX_NAME)

    # 3. Subir los chunks. PineconeVectorStore genera los embeddings y hace
    #    el upsert. El texto original queda guardado en la metadata (text_key),
    #    para recuperarlo sin una segunda consulta.
    embeddingsModel = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = PineconeVectorStore(index=index, embedding=embeddingsModel)

    print("Subiendo chunks a Pinecone (esto genera embeddings vía OpenAI)...")
    vector_store.add_documents(chunks)

    # 4. Verificación.
    stats = index.describe_index_stats()
    print(f"Ingesta completa. Estado del índice: {stats}")


if __name__ == "__main__":
    main()
