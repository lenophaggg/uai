#!/usr/bin/env python
# src/prepare_spacy_data.py

import json
import random
import argparse
import pathlib
import sys

import spacy
from spacy.tokens import DocBin


def convert_to_docbin(examples, nlp):
    """Преобразует список {'text':…, 'entities':[…]} → DocBin."""
    db = DocBin()
    for ex in examples:
        text = ex.get("text")
        ents = ex.get("entities", [])
        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in ents:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                spans.append(span)
        doc.ents = spans
        db.add(doc)
    return db


def main():
    p = argparse.ArgumentParser(
        description="Готовит train.spacy и dev.spacy из ner_dataset.json"
    )
    p.add_argument(
        "--json", "-j", type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] /
                "data" / "raw" / "ner_dataset.json",
        help="Путь к ner_dataset.json (default: %(default)s)"
    )
    p.add_argument(
        "--out", "-o", type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "tmp",
        help="Каталог для .spacy (default: %(default)s)"
    )
    p.add_argument(
        "--split", "-s", type=float, default=0.1,
        help="Доля dev-корпуса (default: 0.1)"
    )
    p.add_argument(
        "--seed",    type=int, default=42,
        help="Сид для перемешивания (default: 42)"
    )
    args = p.parse_args()

    if not args.json.exists():
        sys.exit(f"❌ Файл не найден: {args.json}")

    args.out.mkdir(parents=True, exist_ok=True)
    with args.json.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit(f"❌ Формат ожидается: список объектов, а не {type(data)}")

    random.seed(args.seed)
    random.shuffle(data)
    split_at = int(len(data) * (1 - args.split))
    train_examples = data[:split_at]
    dev_examples   = data[split_at:]

    nlp = spacy.blank("ru")
    train_db = convert_to_docbin(train_examples, nlp)
    dev_db   = convert_to_docbin(dev_examples,   nlp)

    train_file = args.out / "train.spacy"
    dev_file   = args.out / "dev.spacy"
    train_db.to_disk(train_file)
    dev_db.to_disk(dev_file)

    print(f"✔ Train: {len(train_examples)} → {train_file}")
    print(f"✔ Dev:   {len(dev_examples)}  → {dev_file}")


if __name__ == "__main__":
    main()
