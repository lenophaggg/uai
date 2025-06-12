from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

es = Elasticsearch("http://localhost:9200")

SBERT_MODEL_PATH = r"C:\Users\renzh\OneDrive\Desktop\uai\api\sbert_model"
DEFAULT_MODEL = SentenceTransformer(SBERT_MODEL_PATH)

QA_MODEL_PATH = r"C:\Users\renzh\OneDrive\Desktop\uai\qa\models\rubert-tiny2-faq"
QA_MODEL = SentenceTransformer(QA_MODEL_PATH)

ENTITY_TO_FIELD = {
    "PER": "namecontact",
    "ROLE": "position",
    "DEPARTMENT": "position",
    "DEPARTMENT_ABBR": "position",
    "SUBJECT": "taughtsubjects",
    "GROUP": "groups",
}

DEPARTMENT_ABBR_MAP = {
    "фцпт": "Факультет цифровых промышленных технологий",
    "фкэа": "Факультет кораблестроения энергетики и автоматики",
    "фенго": "Факультет естественнонаучного и гуманитарного образования",
}

def _expand_value(val: str) -> List[str]:
    vals = [val]
    lower = val.lower()
    if lower in DEPARTMENT_ABBR_MAP:
        vals.append(DEPARTMENT_ABBR_MAP[lower])
    return vals

def _normalize_filters(filters: Dict[str, List[str]]) -> Dict[str, List[str]]:
    norm_filters = {}
    for field, values in filters.items():
        normed = []
        for v in values:
            lower = v.lower()
            if field == "position" and lower in DEPARTMENT_ABBR_MAP:
                normed.append(DEPARTMENT_ABBR_MAP[lower])
            else:
                normed.append(v)
        norm_filters[field] = normed
    return norm_filters

def normalize_position_abbr(filters: dict, abbr_map: dict) -> dict:
    """Заменяет аббревиатуры факультетов на полные названия в фильтре position."""
    filters = filters.copy()
    if "position" in filters:
        new_values = []
        for v in filters["position"]:
            lower = v.lower()
            if lower in abbr_map:
                new_values.append(abbr_map[lower])
            else:
                new_values.append(v)
        filters["position"] = new_values
    return filters

def passes_ner_filters(contact: Dict[str, Any], filters: Dict[str, List[str]]) -> bool:
    """
    Для каждого значения в фильтре ищем его вхождение в строку соответствующего поля.
    Фильтр пройдён, если все значения из filters встречаются в строке.
    """
    for field, values in filters.items():
        c_val = contact.get(field, "")
        if isinstance(c_val, list):
            c_text = " ".join(map(str, c_val)).lower()
        else:
            c_text = str(c_val).lower()
        # Теперь ищем КАЖДОЕ слово из фильтра по отдельности!
        for v in values:
            # Фильтруем по каждому слову (например, "декан", "факультет", "цифровых", ...)
            for word in v.lower().split():
                if word not in c_text:
                    return False
    return True

def _vector_search(query: str, k: int, index: str, model, fields: List[str]):
    try:
        q_vec = model.encode(query)
        resp = es.search(index=index, body={"size": 1000, "query": {"match_all": {}}, "_source": fields})
        docs = resp["hits"]["hits"]

        results = []
        for hit in docs:
            doc = hit["_source"]
            doc_q = doc.get("q", "")
            doc_vec = model.encode(doc_q)
            score = float(np.dot(q_vec, doc_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(doc_vec) + 1e-8))
            doc["score"] = score
            results.append(doc)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


def vector_search_qa(query: str, k: int = 5) -> Dict[str, Any]:
    faq_results = _vector_search(query, k, "faq_qa", QA_MODEL, ["q", "a"])
    if faq_results and faq_results[0]["score"] > 0.3:
        best = faq_results[0]
        return {"type": "faq", "question": best["q"], "answer": best["a"], "score": best["score"]}
    doc_results = _vector_search(query, k, "documents", DEFAULT_MODEL, ["content", "filename"])
    if doc_results and doc_results[0]["score"] > 0.3:
        best = doc_results[0]
        return {"type": "document", "fragment": best["content"], "filename": best["filename"], "score": best["score"]}
    return {"type": "document", "fragment": None}

def vector_search_ner(
    query: str,
    filters: Dict[str, List[str]],
    k: int = 5
) -> List[Dict[str, Any]]:
    try:
        FIELDS = [
            "namecontact", "position", "taughtsubjects", "groups", "telephone", "email"
        ]

        # 1. Получаем все документы
        resp = es.search(
            index="person_contacts",
            body={"size": 2000, "query": {"match_all": {}}},
            _source=FIELDS
        )
        docs = resp.get("hits", {}).get("hits", [])
        if not docs:
            return []

        # 2. Вектор запроса
        model = DEFAULT_MODEL
        q_vec = model.encode(query)

        candidates = []
        for d in docs:
            src = d.get("_source", {})

            # 3. Формируем текст из всех полей, участвующих в описании
            position = src.get("position", [])
            name = src.get("namecontact", "")
            subjects = src.get("taughtsubjects", [])
            groups = src.get("groups", [])

            doc_text_parts = [
                name,
                " ".join(position) if isinstance(position, list) else str(position),
                " ".join(subjects),
                " ".join(groups)
            ]
            doc_text = " ".join(doc_text_parts)

            d_vec = model.encode(doc_text)
            sim = float((q_vec @ d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec) + 1e-8))

            src["score"] = sim
            candidates.append(src)

        # 4. Сортировка по score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # 5. Фильтрация по порогу (по желанию — можно убрать или ослабить)
        SCORE_THRESHOLD = 0.3
        filtered = [c for c in candidates if c["score"] > SCORE_THRESHOLD]

        return filtered[:k] if filtered else candidates[:k]

    except Exception as e:
        logger.error(f"vector_search_ner failed: {e}")
        return []
