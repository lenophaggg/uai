# src/train_ner.py
import json, random, sys, subprocess, re
from pathlib import Path
import spacy
from spacy.tokens import DocBin
from spacy.language import Language

# ─── ПУТИ ───────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent        # .../ner
DATA_JSON   = ROOT / "data" / "raw" / "ner_dataset.json"
TMP_DIR     = ROOT / "src" / "tmp"
MODEL_DIR   = ROOT / "models" / "spacy_ner"
CONFIG_PATH = TMP_DIR / "ner_config.cfg"

TMP_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─── УТИЛИТЫ ────────────────────────────────────────────────────────────
def load_dataset(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def create_custom_tokenizer(nlp):
    # Define regex for hyphenated words to prevent splitting
    hyphenated_re = re.compile(r"[\wЁёА-Яа-я]+-[\wЁёА-Яа-я]+")
    
    # Get the default tokenizer
    tokenizer = nlp.tokenizer
    
    # Create a custom infix_finditer to preserve hyphenated words
    def custom_infix_finditer(text):
        # Find all infix patterns except within hyphenated words
        for match in re.finditer(r'[\.,!?;\(\)\[\]\{\}]', text):
            if not hyphenated_re.match(text[max(0, match.start()-10):match.end()+10]):
                yield match
    
    # Update the tokenizer with custom infix_finditer
    tokenizer.infix_finditer = custom_infix_finditer
    return tokenizer

def to_docbin(samples, nlp):
    db = DocBin()
    for ex in samples:
        text = ex["text"]
        doc = nlp.make_doc(text)
        bio_labels = ex["bio_labels"]  # Новый формат с BIO-метками
        
        # Преобразуем BIO-метки в сущности для SpaCy
        entities = []
        current_entity = None
        start = 0
        char_pos = 0
        
        for i, (word, label) in enumerate(bio_labels):
            word_start = char_pos
            char_pos += len(word) + 1  # +1 для пробела

            if label.startswith("B-"):
                # Начало новой сущности
                if current_entity:  # Завершаем предыдущую сущность, если она есть
                    entities.append(doc.char_span(current_entity[0], word_start - 1, label=current_entity[1], alignment_mode="contract"))
                current_entity = (word_start, label[2:])  # Сохраняем начало и метку (без "B-")
            
            elif label.startswith("I-") and current_entity:
                # Продолжение текущей сущности
                pass  # Просто продолжаем, конец сущности обработаем на следующем шаге
            
            else:
                # Метка "O" или начало новой сущности без предыдущей
                if current_entity:
                    entities.append(doc.char_span(current_entity[0], word_start - 1, label=current_entity[1], alignment_mode="contract"))
                    current_entity = None

        # Завершаем последнюю сущность, если она есть
        if current_entity:
            entities.append(doc.char_span(current_entity[0], char_pos - 1, label=current_entity[1], alignment_mode="contract"))

        # Фильтруем None значения и устанавливаем сущности
        doc.ents = [span for span in entities if span is not None]
        db.add(doc)
    return db

# ─── ОСНОВНОЙ СЦЕНАРИЙ ─────────────────────────────────────────────────
def main():
    # 1. Подготовка датасета
    data = load_dataset(DATA_JSON)
    random.shuffle(data)
    split = int(0.9 * len(data))
    
    # Create blank Russian language model with custom tokenizer
    nlp_blank = spacy.blank("ru")
    nlp_blank.tokenizer = create_custom_tokenizer(nlp_blank)

    to_docbin(data[:split], nlp_blank).to_disk(TMP_DIR / "train.spacy")
    to_docbin(data[split:], nlp_blank).to_disk(TMP_DIR / "dev.spacy")

    # 2. Генерируем конфиг (если ещё нет)
    if not CONFIG_PATH.exists():
        subprocess.run(
            [sys.executable, "-m", "spacy", "init", "config",
             str(CONFIG_PATH), "--lang", "ru", "--pipeline", "ner",
             "--optimize", "efficiency", "--force"],
            check=True
        )
        # добавляем правильный оптимизатор
        with CONFIG_PATH.open(encoding="utf-8") as f:
            cfg = f.read()
        cfg = cfg.replace(
            "[training]\n", 
            "[training]\n\n[training.optimizer]\n@optimizers = \"Adam.v1\"\nlearn_rate = 0.001\n"
        )
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            f.write(cfg)

    # 3. Обучение
    print(f"\nℹ Saving to output directory: {MODEL_DIR}")
    subprocess.run(
        [sys.executable, "-m", "spacy", "train", str(CONFIG_PATH),
         "--output", str(MODEL_DIR),
         "--paths.train", str(TMP_DIR / "train.spacy"),
         "--paths.dev",   str(TMP_DIR / "dev.spacy"),
         "--gpu-id", "-1"],
        check=True
    )
    print(f"\n✔ Модель сохранена в {MODEL_DIR}")

if __name__ == "__main__":
    main()