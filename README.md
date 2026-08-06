# Pre-entrega 4 — Sistema RAG Escalable en la Nube con Pinecone

Sistema RAG (Retrieval-Augmented Generation) escalable que migra la arquitectura de recuperación de un entorno local a **Pinecone Serverless** (nube), implementa **búsqueda híbrida** (semántica + léxica BM25) y evalúa el rendimiento con métricas de recuperación (**Precision@k** y **Recall@k**).

El dataset de ejemplo son documentos sobre la ciudad de **Mar del Plata** (historia, playas, gastronomía).

## Arquitectura

```
data/*.md
   │
   ▼
loader.py ──────────► chunks (List[Document])
   │                       │
   │                       ├──► ingest.py ──► embeddings (OpenAI) ──► Pinecone (semántico)
   │                       │
   │                       └──► rag_system.py ──► BM25Retriever (léxico, en memoria)
   │                                                     │
   │                          PineconeVectorStore ───────┤
   │                                                     ▼
   │                                          EnsembleRetriever (híbrido, RRF)
   │                                                     │
   │                                                     ▼
   └──────────────────────────────► evaluate.py ──► Precision@5 / Recall@5
```

- **`loader.py`**: módulo compartido. Lee los `.md`, los fragmenta (500 chars / 50 overlap) y asigna el nombre de archivo como `source` en la metadata. Lo usan tanto la ingesta como el BM25, garantizando que ambos retrievers vean los mismos chunks.
- **`ingest.py`**: crea el índice Serverless (si no existe) y sube los chunks a Pinecone generando embeddings con `text-embedding-3-small`.
- **`rag_system.py`**: clase `RAGSystem` que combina el retriever semántico (Pinecone) y el léxico (BM25) en un `EnsembleRetriever`.
- **`evaluate.py`**: calcula Precision@5 y Recall@5 sobre un Golden Set de preguntas con documento fuente conocido.

## Requisitos previos

- Python 3.12
- Una cuenta de [Pinecone](https://www.pinecone.io/) (el free tier alcanza) con una API key.
- Créditos cargados en [platform.openai.com](https://platform.openai.com/) (la ingesta y las consultas generan embeddings, que son de pago).

## Instalación (macOS / zsh)

```zsh
# 1. Crear y activar un entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar las dependencias
pip install -r requirements.txt

# 3. Configurar las credenciales
cp .env.example .env
# Editá .env con tus claves reales de Pinecone y OpenAI
```

Explicación de los comandos:

- `python3 -m venv .venv` — crea un entorno virtual aislado en la carpeta `.venv` (equivalente conceptual a un `node_modules` local del proyecto).
- `source .venv/bin/activate` — activa el entorno; a partir de acá `pip` y `python` usan las versiones del proyecto. Para salir: `deactivate`.
- `cp .env.example .env` — copia la plantilla; luego se completa con las claves reales. El `.env` está en `.gitignore` y nunca se sube.

## Uso

Ejecutar **desde la raíz del proyecto** (no desde `src/`), en este orden:

```zsh
# 1. Ingestar los documentos a Pinecone (genera embeddings — consume créditos OpenAI)
python src/ingest.py

# 2. (Opcional) Probar una consulta al recuperador híbrido
python src/rag_system.py

# 3. Evaluar el sistema con métricas Precision@5 / Recall@5
python src/evaluate.py
```

> **Nota de costos:** cada corrida de `ingest.py` regenera los embeddings de todos los chunks. Con este dataset son centavos, pero evitá re-ingestar sin necesidad. El script reutiliza el índice si ya existe.

## Estructura del proyecto

```
coderhouse-ia-preentrega4/
├── src/
│   ├── loader.py          # carga + chunking compartido
│   ├── ingest.py          # ingesta a Pinecone
│   ├── rag_system.py      # recuperador híbrido (Pinecone + BM25)
│   ├── evaluate.py        # métricas Precision@5 / Recall@5
│   └── golden_set.json    # benchmark de preguntas
├── data/                  # documentos .md sobre Mar del Plata
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Decisiones de diseño

- **Búsqueda híbrida (Pinecone + BM25):** la búsqueda semántica capta significado pero falla con términos exactos (fechas, nombres propios como "Havanna"). BM25 rescata esos casos por coincidencia léxica. El `EnsembleRetriever` fusiona ambos con Reciprocal Rank Fusion, pesados 50/50.
- **`loader.py` compartido:** `BM25Retriever` construye su índice en memoria desde documentos locales (no lee de Pinecone). Extraer la carga a un módulo común evita duplicar el chunking y asegura consistencia entre ambos retrievers.
- **`source` como identificador:** el nombre de cada archivo `.md` se guarda como `source` en la metadata de cada chunk, sirviendo de `documento_id` para la evaluación.
- **Golden Set mixto:** las preguntas combinan casos léxicos (fechas, marcas) y semánticos (conceptos), para verificar que el sistema híbrido cubre ambos tipos de consulta.
