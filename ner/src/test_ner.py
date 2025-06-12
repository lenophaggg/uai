import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer
from pathlib import Path

MODEL_DIR = Path("models") / "spacy_ner" / "model-best"
DEV_SPACY = Path("src") / "tmp" / "dev.spacy"

def evaluate_model(nlp, dev_path):
    if not dev_path.exists():
        raise FileNotFoundError(f"Файл {dev_path} не найден.")
    doc_bin = DocBin().from_disk(dev_path)
    examples = []
    total_examples = 0
    total_gold_ents = 0
    total_pred_ents = 0
    for gold in doc_bin.get_docs(nlp.vocab):
        pred = nlp(gold.text)
        examples.append(Example(pred, gold))
        total_examples += 1
        total_gold_ents += len(gold.ents)
        total_pred_ents += len(pred.ents)
    if not examples:
        print("=== Evaluation on dev.spacy ===")
        print("Нет примеров для оценки.")
        return None
    scores = Scorer().score(examples)
    print("=== Evaluation on dev.spacy ===")
    print(f"Общее количество примеров: {total_examples}")
    print(f"Общее количество эталонных сущностей: {total_gold_ents}")
    print(f"Общее количество предсказанных сущностей: {total_pred_ents}")
    precision = scores.get('ents_p')
    recall = scores.get('ents_r')
    f1 = scores.get('ents_f')
    if precision is not None and recall is not None and f1 is not None:
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-score:  {f1:.3f}")
    else:
        print("Невозможно вычислить метрики.")

    return scores

def predict_examples(nlp, sample_texts):
    print("\n=== Sample predictions ===")
    for text in sample_texts:
        doc = nlp(text)
        print(f"\nOriginal:    {text}")
        print(f"Tokens:      {[(token.text, token.idx) for token in doc]}")
        if doc.ents:
            for ent in doc.ents:
                print(f"  • {ent.text} [{ent.start_char}-{ent.end_char}] → {ent.label_}")
        else:
            print("  (no entities)")

if __name__ == "__main__":
    try:
        print(f"Loading model from {MODEL_DIR} …")
        nlp = spacy.load(MODEL_DIR)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        exit(1)

    # 1) Оценка на dev
    try:
        evaluate_model(nlp, DEV_SPACY)
    except Exception as e:
        print(f"Ошибка при оценке модели: {e}")
        exit(1)

    # 2) Примеры для ручной проверки
    sample_texts = [
        "кто ведет сопромат у группы 82016",
        "кто заведующей кафедрой вычислительной техники",
        "кто заведующей квс",
        "кто декан фцпт",
        "кто такая соколова софья",
        "кто такой преподаватель серебряков михаил",
        "контакты колледжа спбгмту",
        "преподаватель по информатике",
        "кто декан",
        "декан",
        "кто колледж",
        "колледж"
    ]
    predict_examples(nlp, sample_texts)

    # 3) Проверка ручного ввода
    print("\nВведи свой пример (или Enter для выхода):")
    while True:
        user_input = input("> ")
        if not user_input.strip():
            break
        predict_examples(nlp, [user_input])
