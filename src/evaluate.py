"""
evaluate.py - Evaluación cuantitativa del recuperador.

Sobre un "Golden Set" (preguntas cuyo documento fuente correcto conocemos de
antemano), mide dos métricas por cada pregunta y promedia:

  - Recall@k:    ¿está el documento correcto entre los k recuperados?
                 Responde: "¿me estoy perdiendo lo relevante?"
  - Precision@k: ¿qué proporción de los k recuperados son del documento correcto?
                 Responde: "¿cuánta basura traigo?"

Uso:
    python src/evaluate.py
"""

import json
from pathlib import Path

from rag_system import RAGSystem, TOP_K


def load_golden_set(path: str = "src/golden_set.json") -> list[dict]:
    """Carga el benchmark de preguntas desde el JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate():
    golden_set = load_golden_set()
    rag = RAGSystem()

    total_recall = 0.0
    total_precision = 0.0

    print(f"Evaluando {len(golden_set)} preguntas (k={TOP_K})...\n")

    for item in golden_set:
        pregunta = item["pregunta"]
        esperado = item["documento_id_esperado"]

        # Recuperar los top-k documentos para la pregunta.
        resultados = rag.retrieve(pregunta)
        fuentes = [doc.metadata.get("source") for doc in resultados]

        # Recall@k: 1 si el documento esperado aparece entre los recuperados, 0 si no.
        hit = esperado in fuentes
        recall = 1.0 if hit else 0.0

        # Precision@k: proporción de los recuperados que son del documento esperado.
        # (cuántos de los k pertenecen a la fuente correcta / total recuperado)
        aciertos = fuentes.count(esperado)
        precision = aciertos / len(fuentes) if fuentes else 0.0

        total_recall += recall
        total_precision += precision

        estado = "OK " if hit else "MISS"
        print(f"[{estado}] {pregunta}")
        print(f"        esperado: '{esperado}' | recuperados: {fuentes}")
        print(f"        recall={recall:.2f}  precision={precision:.2f}\n")

    # Promediar sobre todas las preguntas.
    n = len(golden_set)
    avg_recall = total_recall / n
    avg_precision = total_precision / n

    print("=" * 50)
    print("RESUMEN DE EVALUACIÓN")
    print(f"  Recall@{TOP_K}    promedio: {avg_recall:.2%}")
    print(f"  Precision@{TOP_K} promedio: {avg_precision:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate()
