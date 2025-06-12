#!/usr/bin/env python
"""
CLI-просмотр содержимого индекса FAQ (`faq_qa`).

$ python read_faq.py                 # показать все вопросы
$ python read_faq.py "пропуск"       # найти вопросы, где встречается слово «пропуск»
"""

from __future__ import annotations
import sys
from elasticsearch import Elasticsearch

ES_HOST    = "http://localhost:9200"
INDEX_NAME = "faq_qa"

es = Elasticsearch(ES_HOST)


def list_faq(query_text: str | None = None, size: int = 100) -> None:
    """
    Если query_text == None или "", выводит *все* записи (до size штук).
    Иначе выполняет full-text-поиск по полю 'q'.
    """
    if query_text:
        es_query = {
            "query": {
                "match": {
                    "q": {
                        "query": query_text,
                        "operator": "and"           # можно заменить на "or"
                    }
                }
            },
            "size": size,
            "sort": [
                {"_score": {"order": "desc"}}
            ]
        }
    else:
        es_query = {
            "query": {"match_all": {}},
            "size": size,
            "sort": [
                {"q.keyword": {"order": "asc"}}
            ]
        }

    resp = es.search(index=INDEX_NAME, body=es_query)
    hits = resp["hits"]["hits"]

    if not hits:
        print("⛔   Ничего не найдено.")
        return

    print(f"Найдено {len(hits)} записей (показаны первые {size}):")
    print("-" * 60)
    for i, hit in enumerate(hits, 1):
        source = hit["_source"]
        print(f"{i:>2}.  {source['q']}")
        print(f"    ↳ {source['a']}\n")
    print("-" * 60)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    list_faq(query_text=query or None)
