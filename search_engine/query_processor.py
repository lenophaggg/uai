import joblib
import spacy
from sentence_transformers import SentenceTransformer
from search_engine.vector_search import (
    vector_search_qa, vector_search_ner,
    ENTITY_TO_FIELD, DEPARTMENT_ABBR_MAP, _normalize_filters, normalize_position_abbr
)

ROUTER_MODEL_PATH = r"C:\Users\renzh\OneDrive\Desktop\uai\classifier\models\router_clf.joblib"
NER_MODEL_PATH = r"C:\Users\renzh\OneDrive\Desktop\uai\ner\models\spacy_ner\model-best"
QA_MODEL_PATH = r"C:\Users\renzh\OneDrive\Desktop\uai\qa\models\rubert-tiny2-faq"
SBERT_MODEL_PATH = r"C:\Users\renzh\OneDrive\Desktop\uai\api\sbert_model"

router_clf = joblib.load(ROUTER_MODEL_PATH)
ner_model = spacy.load(NER_MODEL_PATH)
qa_model = SentenceTransformer(QA_MODEL_PATH)
sbert_model = SentenceTransformer(SBERT_MODEL_PATH)

def get_filters_from_ner(query: str):
    doc = ner_model(query)
    filters = {}
    found_entities = []
    for ent in doc.ents:
        field = ENTITY_TO_FIELD.get(ent.label_)
        found_entities.append({"text": ent.text, "label": ent.label_})
        if field:
            filters.setdefault(field, []).append(ent.text)
    return filters, found_entities

def process_query(query: str, k=5):
    route = str(router_clf.predict([query])[0]).strip().lower()

    if route == "ner":
        filters, ner_entities = get_filters_from_ner(query)
        filters_norm = normalize_position_abbr(filters, DEPARTMENT_ABBR_MAP)
        results = vector_search_ner(query, filters=filters_norm, k=k)
        return {
            "type": "contact",
            "filters": _normalize_filters(filters_norm),
            "ner_entities": ner_entities,
            "result": results[0] if results else None
        }
    elif route in ("faq", "qa"):
        return vector_search_qa(query, k=k)
    else:
        return {"type": "unknown", "result": None}

def reload_models():
    global router_clf, ner_model, qa_model, sbert_model
    router_clf = joblib.load(ROUTER_MODEL_PATH)
    ner_model = spacy.load(NER_MODEL_PATH)
    qa_model = SentenceTransformer(QA_MODEL_PATH)
    sbert_model = SentenceTransformer(SBERT_MODEL_PATH)
