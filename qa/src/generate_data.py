import json
from pathlib import Path
import pandas as pd
import numpy as np

FAQ_PATH = Path(r"C:\Users\renzh\OneDrive\Desktop\uai\qa\data\raw\faq_university.json")
SBERQUAD_DIR = Path(r"C:\Users\renzh\OneDrive\Desktop\uai\qa\data\raw\sberquad\sberquad")
OUT_PATH = Path(r"C:\Users\renzh\OneDrive\Desktop\uai\qa\data\processed\sberquad_faq.json")

SBERQUAD_PARQUETS = [
    SBERQUAD_DIR / "train-00000-of-00001.parquet",
    SBERQUAD_DIR / "validation-00000-of-00001.parquet",
    SBERQUAD_DIR / "test-00000-of-00001.parquet"
]

def expand_faq_questions(faq_list):
    expanded = []
    for item in faq_list:
        questions = item.get("questions") or [item.get("question")]
        answer = item["answer"]
        context = item["context"]
        answer_start = item["answer_start"]
        for q in questions:
            expanded.append({
                "id": f"{abs(hash(q + context))}",
                "question": q,
                "context": context,
                "answers": {
                    "text": [answer],
                    "answer_start": [answer_start]
                }
            })
    return expanded

def recursive_to_builtin(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: recursive_to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_to_builtin(x) for x in obj]
    return obj

def load_sberquad_examples(parquet_path):
    examples = []
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            q = row['question']
            context = row['context']
            answers = row['answers']
            answer_texts = answers['text'] if isinstance(answers['text'], (list, np.ndarray)) else [answers['text']]
            answer_starts = answers['answer_start'] if isinstance(answers['answer_start'], (list, np.ndarray)) else [answers['answer_start']]
            examples.append({
                "id": f"{abs(hash(q + context))}",
                "question": q,
                "context": context,
                "answers": {
                    "text": list(answer_texts),
                    "answer_start": list(answer_starts)
                }
            })
    return examples

def main():
    with FAQ_PATH.open(encoding="utf-8") as f:
        faq = json.load(f)
    expanded = expand_faq_questions(faq)

    sber_examples = []
    for parquet_path in SBERQUAD_PARQUETS:
        sber_examples.extend(load_sberquad_examples(parquet_path))

    all_examples = expanded + sber_examples
    all_examples_clean = recursive_to_builtin(all_examples)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_examples_clean, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_examples_clean)} records to {OUT_PATH}")

if __name__ == "__main__":
    main()
