# classifier/src/generate_data.py

import json
import random
import re
from pathlib import Path

BASE = Path(__file__).parent.parent.parent  # .../uai
SQL_PATH = BASE / "data" / "raw" / "interactiveboardentities.sql"
SBER_PATHS = [
    BASE / "data" / "raw" / "sberquad" / "sberquad" / "train-00000-of-00001.parquet",
    BASE / "data" / "raw" / "sberquad" / "sberquad" / "validation-00000-of-00001.parquet",
    BASE / "data" / "raw" / "sberquad" / "sberquad" / "test-00000-of-00001.parquet",
]
DATA_DIR = BASE / "classifier" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Генерируем сырые ner/qa-запросы (router_data.json)
question_starts = [
    "кто ведет", "кто преподает", "кто читает", "кто отвечает за", 
    "кто является", "кто такой", "", "декан", "преподаватель", "группа", 
    "предмет", "контакты", "адрес", "электронная почта"
]

sql = SQL_PATH.read_text(encoding="utf-8")

faculties = re.findall(r"COPY public\.faculties \(name\) FROM stdin;\n(.*?)\\\.", sql, re.DOTALL)
faculties = faculties[0].split("\n") if faculties else []

groups = re.findall(r"COPY public\.groups \(groupnumber, facultyname\) FROM stdin;\n(.*?)\\\.", sql, re.DOTALL)
groups = [line.split("\t")[0] for line in groups[0].split("\n") if line.strip()] if groups else []

names = re.findall(r"\d+\t([А-ЯЁа-яё\- ]+)\t\{", sql)

subjects = re.findall(r"COPY public\.subjects \(subjectname\) FROM stdin;\n(.*?)\\\.", sql, re.DOTALL)
subjects = subjects[0].split("\n") if subjects else []

ner_queries = []
for faculty in faculties:
    for start in question_starts:
        ner_queries.append(f"{start} {faculty}".strip())
for group in groups:
    for start in question_starts:
        ner_queries.append(f"{start} группа {group}".strip())
for name in names:
    for start in question_starts:
        ner_queries.append(f"{start} {name}".strip())
for subj in subjects:
    for start in question_starts:
        ner_queries.append(f"{start} {subj}".strip())

# QA-запросы (базовые)
qa_queries = [
    "когда начинается сессия",
    "где взять справку",
    "какой график работы библиотеки",
    "как найти расписание",
    "какие сегодня пары",
    "что задали по математике",
    "когда выходной",
    "где находится деканат",
    "как записаться на консультацию",
    "как получить студенческий билет",
    "какие факультеты есть в вузе",
    "какая стипендия назначается первокурсникам",
    "как оплатить обучение",
    "где найти расписание экзаменов",
    "как восстановиться после отчисления",
    "кто мой куратор",
    "где можно поесть в университете",
    "когда следующий учебный семестр",
    "как получить справку-вызов",
    "какие языки преподают в вузе"
]

# Сохраняем сырой датасет (без балансировки)
router_data = [{"query": q.strip(), "label": "ner"} for q in ner_queries] + [{"query": q.strip(), "label": "qa"} for q in qa_queries]
random.shuffle(router_data)

router_data_path = DATA_DIR / "router_data.json"
with open(router_data_path, "w", encoding="utf-8") as f:
    json.dump(router_data, f, ensure_ascii=False, indent=2)
print(f"✔ Сохранил {len(router_data)} записей в {router_data_path}")

# 2. Готовим балансированный датасет для обучения роутера (router_balanced_data.json)
ner_data = [x for x in router_data if x["label"] == "ner"]

# Дополняем qa запросами из SberQuAD (если есть)
sber_questions = []
try:
    import pandas as pd
    for path in SBER_PATHS:
        if path.exists():
            df = pd.read_parquet(path)
            sber_questions.extend(df["question"].dropna().tolist())
    if sber_questions:
        qa_queries += random.sample(sber_questions, min(3000, len(sber_questions)))
except Exception as e:
    print(f"[!] Не удалось загрузить SberQuAD: {e}")

qa_data = [{"query": q.strip(), "label": "qa"} for q in qa_queries if q.strip()]
qa_count = max(int(0.4 * len(ner_data)), 1)
if len(qa_data) < qa_count:
    qa_data = (qa_data * ((qa_count + len(qa_data) - 1) // len(qa_data)))[:qa_count]
else:
    qa_data = random.sample(qa_data, qa_count)

router_balanced = ner_data + qa_data
random.shuffle(router_balanced)

router_balanced_path = DATA_DIR / "router_balanced_data.json"
with open(router_balanced_path, "w", encoding="utf-8") as f:
    json.dump(router_balanced, f, ensure_ascii=False, indent=2)
print(f"✔ Сохранил {len(router_balanced)} записей (qa: {len(qa_data)}, ner: {len(ner_data)}) в {router_balanced_path}")
