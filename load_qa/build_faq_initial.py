#!/usr/bin/env python
"""
Создаёт индекс `faq_qa` и загружает FAQ-пары.
Запуск:  python build_faq_initial.py
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

ES_HOST = "http://localhost:9200"
INDEX   = "faq_qa"

# ✔ указываем абсолютный путь к модели
MODEL_DIR = Path(r"C:\Users\renzh\OneDrive\Desktop\uai\api\sbert_model")

faq_pairs: List[Dict[str, str]] = [
    {"q": "что делать при потере зачётной книжки или студбилета?",
     "a": "Заполнить заявление в деканате и ждать 5 рабочих дней для восстановления."},
    {"q": "что делать при пропуске занятий?",
     "a": "Заполнить заявление в деканате."},
    {"q": "сколько дней делается справка об обучении и как её сделать?",
     "a": "Подайте заявление в ИСУ → «Заявления» → «Студенческий отдел кадров». "
           "Справка будет готова в деканате через 5 рабочих дней."},
    {"q": "какие сроки закрытия задолженностей?",
     "a": "Нужно обратиться в деканат."},
    {"q": "как перевестись на факультет?",
     "a": "Переводы осуществляются в течение 30 дней после начала семестра. "
           "Обратитесь в деканат."},
    {"q": "как взять академический отпуск?",
     "a": "Нужно обратиться в деканат с подтверждающими документами."},
    {"q": "где узнать электронную почту деканата?",
     "a": "decanatfdit@smtu.ru"},
    {"q": "что делать при потере пропуска?",
     "a": "Заполнить заявление в деканате и поехать на Лоцманскую в бюро пропусков."},
    {"q": "как узнать про практику?",
     "a": "Позвонить в отдел практики."},
    {"q": "как решить вопрос с целевым обучением?",
     "a": "Позвонить в отдел целевого обучения."},
    {"q": "как закрыть задолженность по дисциплине?",
     "a": "Взять разрешение в деканате и вместе с ним идти к преподавателю."},
    {"q": "как вернуть деньги за проезд домой во время каникул?",
     "a": "Обратитесь в стипендиальный отдел (https://vk.com/stipendiasmtu)."},
]


def ensure_index(es: Elasticsearch, dims: int) -> None:
    """Создаём индекс, если он ещё не существует."""
    if es.indices.exists(index=INDEX):
        return

    es.indices.create(
        index=INDEX,
        body={
            "mappings": {
                "properties": {
                    "q": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        },
                    },
                    "a": {"type": "text"},
                    "q_vec": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        },
    )
    print(f"Created index '{INDEX}' (vector dim = {dims}).")


def main() -> None:
    if not MODEL_DIR.exists():
        raise SystemExit(f"❌  SBERT-модель не найдена: {MODEL_DIR}")

    model = SentenceTransformer(str(MODEL_DIR))
    dims  = model.get_sentence_embedding_dimension()

    es = Elasticsearch(ES_HOST)
    ensure_index(es, dims)

    actions = [
        {
            "_op_type": "index",
            "_index": INDEX,
            "_source": {
                "q": item["q"],
                "a": item["a"],
                "q_vec": model.encode(item["q"]).tolist(),
            },
        }
        for item in faq_pairs
    ]

    helpers.bulk(es, actions, request_timeout=120)
    print(f"Ingested {len(actions)} FAQ pairs into '{INDEX}'.")


if __name__ == "__main__":
    main()
