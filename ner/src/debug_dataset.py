import json, pathlib, spacy, collections, itertools as it
DATA = pathlib.Path("data/raw/ner_dataset.json")
nlp  = spacy.blank("ru")

stats = collections.Counter()
with DATA.open(encoding="utf-8") as f:
    for ex in json.load(f):
        doc = nlp.make_doc(ex["text"])
        for s, e, lbl in ex.get("entities", []):        # ← имя поля!
            span = doc.char_span(s, e, label=lbl, alignment_mode="contract")
            stats["total_spans"] += 1
            if span is None:
                stats["bad_spans"] += 1
            else:
                stats[f"label::{lbl}"] += 1

print("Всего аннотаций:", stats["total_spans"])
print("Не совпали индексы (char_span is None):", stats["bad_spans"])
print("Распределение меток:",
      {k[7:]: v for k, v in stats.items() if k.startswith("label::")})
