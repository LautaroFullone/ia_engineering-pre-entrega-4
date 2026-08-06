"""
loader.py - Módulo de carga y fragmentación compartido.

Lee los .md de la carpeta /data, los parte en chunks y devuelve una lista de
objetos Document. Esta lógica es COMPARTIDA por dos consumidores:
  - ingest.py    -> sube estos chunks a Pinecone (retriever semántico).
  - rag_system.py -> alimenta con estos chunks al BM25Retriever (retriever léxico).

Se extrae acá para no duplicar el chunking y garantizar que Pinecone y BM25
vean EXACTAMENTE los mismos fragmentos (si difirieran, la búsqueda híbrida
daría resultados inconsistentes).
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_and_chunk(data_dir: str = "data") -> list[Document]:
    """
    Lee todos los .md de 'data_dir', les asigna su nombre de archivo como
    'source' en la metadata, y los fragmenta en chunks de 500 caracteres
    con 50 de overlap.

    El campo 'source' (ej: "historia", "playas") es el documento_id que el
    Golden Set usa para evaluar Precision/Recall.

    Returns:
        Lista de chunks (Document), cada uno con su 'source' en la metadata.
    """
    documents = []

    # 1. Leer cada .md de la carpeta /data.
    for md_file in Path(data_dir).glob("*.md"):
        loader = TextLoader(str(md_file), encoding="utf-8")
        raw_docs = loader.load()

        # Guardar el nombre del archivo (sin extensión) como 'source'.
        for doc in raw_docs:
            doc.metadata["source"] = md_file.stem  # "historia.md" -> "historia"

        documents.extend(raw_docs)

    # 2. Chunkear preservando la metadata (cada chunk hereda el 'source').
    #    NOTA: el splitter cuenta CARACTERES, no tokens.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    # Prueba rápida: correr `python src/loader.py` desde la raíz del proyecto
    # para ver cuántos chunks se generan y de qué fuentes.
    chunks = load_and_chunk()
    print(f"Total de chunks generados: {len(chunks)}")
    fuentes = {c.metadata["source"] for c in chunks}
    print(f"Fuentes encontradas: {fuentes}")
