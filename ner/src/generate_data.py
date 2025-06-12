import json
import random
import re
import itertools
from pathlib import Path
from functools import lru_cache
import pymorphy3

BASE = Path(r"C:\Users\renzh\OneDrive\Desktop\uai\ner")
DATA_DIR = BASE / "data"
RAW_DIR = DATA_DIR / "raw"
DEST_PATH = RAW_DIR / "ner_dataset.json"
SQL_PATH = RAW_DIR / "interactiveboardentities.sql"
TOTAL_SAMPLES = 30000
RAW_DIR.mkdir(parents=True, exist_ok=True)

morph = pymorphy3.MorphAnalyzer()

@lru_cache(maxsize=50000)
def _inflect_word(word: str, case: str) -> str:
    p = morph.parse(word)[0]
    f = p.inflect({case})
    return f.word if f else word

@lru_cache(maxsize=20000)
def inflect_phrase(phrase: str, case: str) -> str:
    return " ".join(_inflect_word(w, case) for w in phrase.split())

def extract_from_copy(sql_text, table_name, columns):
    columns_str = ', '.join(columns)
    pattern = rf"COPY public\.{table_name} \({columns_str}\) FROM stdin;\n(.*?)\n\\\."
    match = re.search(pattern, sql_text, re.DOTALL)
    if match:
        lines = match.group(1).splitlines()
        data = [line.split('\t') for line in lines if line.strip()]
        return data
    return []

def extract_departments(position_str):
    departments = []
    positions = re.findall(r'"(.*?)"', position_str)
    for pos in positions:
        match = re.search(r'Кафедра (.+?)(?=\))', pos)
        if match:
            dept = match.group(1).strip()
            departments.append(dept)
    return departments

with open(SQL_PATH, 'r', encoding='utf-8') as f:
    sql_text = f.read()

groups_data = extract_from_copy(sql_text, "groups", ["groupnumber", "facultyname"])
faculty_names = [row[1] for row in groups_data if row[1] != '\\N']
group_numbers = [row[0] for row in groups_data if row[0] != '\\N']

subjects_data = extract_from_copy(sql_text, "person_taughtsubjects", ["idcontact", "subjectname"])
subject_names = [row[1].strip() for row in subjects_data]

contacts_columns = [
    "idcontact", "namecontact", "\"position\"", "academicdegree", "teachingexperience",
    "telephone", "email", "information", "imgpath", "university_idcontact"
]
contacts_data = extract_from_copy(sql_text, "person_contacts", contacts_columns)
names = [row[1].strip() for row in contacts_data]

departments = []
department_abbrs = set()
for row in contacts_data:
    position_str = row[2]
    depts = extract_departments(position_str)
    departments.extend(depts)
    # Собираем сокращения кафедр, если есть
    for dept in depts:
        abbr = "".join([w[0] for w in dept.split() if w[0].isalpha()]).lower()
        if len(abbr) >= 2 and len(abbr) <= 6:
            department_abbrs.add(abbr)

ORGANIZATIONS = sorted(set(faculty_names))
DEPARTMENTS = sorted(set(departments))
DEPARTMENT_ABBR = sorted(department_abbrs)
SUBJECTS = sorted(set(subject_names))
NAMES = sorted(set(names))
GROUPS = sorted(set(group_numbers))
ROLES = [
    "декан", "заведующий", "заведующая", "преподаватель", "ассистент", "доцент", "профессор",
    "лаборант", "инженер", "старший преподаватель"
]

MANUAL_ORG_ABBR = [
    "ФМП", "ФКЭА", "ФКО", "ИЭФ", "ФЕНГО", "ФЦПТ", "СТФ"
]

GEN_MAP = {
    "ORG": ORGANIZATIONS + MANUAL_ORG_ABBR,
    "DEPARTMENT": DEPARTMENTS,
    "DEPARTMENT_ABBR": DEPARTMENT_ABBR,
    "SUBJECT": SUBJECTS,
    "NAME": NAMES,
    "ROLE": ROLES,
    "GROUP": GROUPS,
}

LABEL_MAP = {
    "ORG": "ORG",
    "DEPARTMENT": "DEPARTMENT",
    "DEPARTMENT_ABBR": "DEPARTMENT",
    "SUBJECT": "SUBJECT",
    "NAME": "NAME",
    "ROLE": "ROLE",
    "GROUP": "GROUP"
}

INTRO_PHRASES = [
    "пожалуйста", "подскажи", "а", "извини", "можешь сказать", "не знаешь"
]

# Падежи
DEFAULT_CASES = {
    "NAME": ["nomn", "gent", "ablt"],
    "ROLE": ["nomn", "gent", "ablt"],
    "DEPARTMENT": ["nomn", "gent", "ablt"],
    "DEPARTMENT_ABBR": [None],
    "ORG": ["nomn", "gent", "ablt"],
    "SUBJECT": ["nomn", "gent", "datv", "accs", "ablt"],
    "GROUP": ["nomn", "gent", "ablt"],
}

# Шаблоны для примеров с правильными границами
TEMPLATE_SCHEMAS = [
    # SUBJECT + GROUP (сложные запросы)
    ("кто ведет {SUBJECT} у группы {GROUP}", [("SUBJECT", None), ("GROUP", None)]),
    ("кто преподает {SUBJECT} в группе {GROUP}", [("SUBJECT", None), ("GROUP", None)]),
    # ROLE + DEPARTMENT/DEPARTMENT_ABBR
    ("кто {ROLE} {DEPARTMENT}", [("ROLE", None), ("DEPARTMENT", None)]),
    ("кто {ROLE} {DEPARTMENT_ABBR}", [("ROLE", None), ("DEPARTMENT_ABBR", None)]),
    # ROLE + ORG
    ("кто {ROLE} {ORG}", [("ROLE", None), ("ORG", None)]),
    # ФИО
    ("кто такая {NAME}", [("NAME", None)]),
    ("кто такой {ROLE} {NAME}", [("ROLE", None), ("NAME", None)]),
    # ORG и DEPARTMENT отдельные
    ("контакты {ORG}", [("ORG", None)]),
    ("контакты {DEPARTMENT}", [("DEPARTMENT", None)]),
    # Короткие одиночные
    ("декан", [("ROLE", None)]),
    ("колледж", [("ORG", None)]),
    ("{DEPARTMENT_ABBR}", [("DEPARTMENT_ABBR", None)]),
    # DEPARTMENT в именительном падеже отдельно (кафедра)
    ("кафедра {DEPARTMENT}", [("DEPARTMENT", None)]),
    # ROLE + SUBJECT
    ("преподаватель по {SUBJECT}", [("ROLE", None), ("SUBJECT", None)]),
    # Только SUBJECT
    ("{SUBJECT}", [("SUBJECT", None)]),
    # Имя
    ("{NAME}", [("NAME", None)]),
    # GROUP
    ("группа {GROUP}", [("GROUP", None)]),
    ("{GROUP}", [("GROUP", None)]),
]

# Негативные шаблоны
NEGATIVE_TEMPLATES = [
    {"text": "какая погода сегодня", "ents": []},
    {"text": "кто расписание", "ents": []},
    {"text": "список студентов", "ents": []},
    {"text": "вывести информацию", "ents": []},
    {"text": "контакты студентов", "ents": []},
    {"text": "завести кафедру", "ents": []},
    {"text": "преподаватель знает математику", "ents": []},
    {"text": "где находится здание", "ents": []},
    {"text": "у кого занятие по информатике", "ents": []},
    {"text": "найди телефоны", "ents": []},
]

def inflect_sample(key, case):
    if key in ("DEPARTMENT_ABBR", "ORG"):
        return random.choice(GEN_MAP[key])
    elif key in GEN_MAP and GEN_MAP[key]:
        return inflect_phrase(random.choice(GEN_MAP[key]), case)
    else:
        return f"UNKNOWN_{key}"

def random_intro():
    return random.choice(INTRO_PHRASES)

# BIO-токенизация для разметки
def bio_tokenize(text, entities):
    tokens = []
    labels = []
    idx = 0
    entity_map = {}
    for start, end, label in entities:
        for i in range(start, end):
            entity_map[i] = (start, end, label)
    for match in re.finditer(r'\S+', text):
        token = match.group()
        t_start = match.start()
        t_end = match.end()
        if t_start in entity_map:
            start, end, label = entity_map[t_start]
            prefix = "B-"
            # Проверяем, входит ли токен полностью в этот спан
            if t_end > end:
                prefix = "B-"
            elif t_start > start:
                prefix = "I-"
            labels.append(f"{prefix}{label}")
        else:
            labels.append("O")
        tokens.append(token)
    return list(zip(tokens, labels))

# --- Генератор опечаток ---
RUSSIAN_ALPHABET = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'

def typo(word):
    if len(word) < 3 or random.random() > 0.25:
        return word
    kind = random.choice(['swap', 'del', 'replace'])
    i = random.randint(0, len(word)-2) if len(word) > 2 else 0
    if kind == 'swap' and len(word) > 3:
        return word[:i] + word[i+1] + word[i] + word[i+2:]
    elif kind == 'del':
        return word[:i] + word[i+1:]
    elif kind == 'replace':
        return word[:i] + random.choice(RUSSIAN_ALPHABET) + word[i+1:]
    return word

def maybe_typos(tokens_and_labels, prob=0.15):
    out = []
    for token, label in tokens_and_labels:
        if token.isalpha() and random.random() < prob:
            out.append([typo(token), label])
        else:
            out.append([token, label])
    return out

# Автоматически генерируем TEMPLATES с разными падежами
TEMPLATES = []
for schema in TEMPLATE_SCHEMAS:
    tpl_text, ents_schema = schema
    cases_list = []
    for key, case in ents_schema:
        cases_list.append(DEFAULT_CASES.get(key, [None]))
    for case_combo in itertools.product(*cases_list):
        ents = [(k, c) for (k, _), c in zip(ents_schema, case_combo)]
        TEMPLATES.append({"text": tpl_text, "ents": ents})

# Генерация одного примера (с опечатками!)
def generate_sample():
    # С вероятностью 10% — негативный пример
    if random.random() < 0.1:
        tpl = random.choice(NEGATIVE_TEMPLATES)
        text_out = tpl["text"]
        tokens = text_out.split()
        bio = [[token, "O"] for token in tokens]
        bio = maybe_typos(bio, prob=0.2)
        return {"text": " ".join([tok for tok, _ in bio]), "bio_labels": bio}

    tpl = random.choice(TEMPLATES)
    text = tpl["text"]
    ents = tpl["ents"]
    # Вставляем интро-фразу если встречается плейсхолдер
    if "{INTRO}" in text:
        text = text.replace("{INTRO}", random_intro())

    # Подстановка сущностей по шаблону
    entity_spans = []
    text_out = ""
    last_idx = 0
    for ent in ents:
        key, case = ent
        pholder = "{" + key + "}"
        before = text.find(pholder, last_idx)
        if before == -1:
            continue
        text_out += text[last_idx:before]
        val = inflect_sample(key, case) if case else inflect_sample(key, "nomn")
        start = len(text_out)
        text_out += val
        end = len(text_out)
        entity_spans.append((start, end, LABEL_MAP[key]))
        last_idx = before + len(pholder)
    text_out += text[last_idx:]
    # BIO разметка
    tokens_and_labels = []
    pos = 0
    token_regex = re.compile(r'\S+')
    entity_ptr = 0
    current_entity = entity_spans[entity_ptr] if entity_spans else None
    cur_start, cur_end, cur_label = current_entity if current_entity else (None, None, None)
    for match in token_regex.finditer(text_out):
        token = match.group()
        t_start = match.start()
        t_end = match.end()
        if current_entity and t_start >= cur_start and t_end <= cur_end:
            prefix = "B-" if t_start == cur_start else "I-"
            tokens_and_labels.append([token, f"{prefix}{cur_label}"])
            if t_end == cur_end and entity_ptr + 1 < len(entity_spans):
                entity_ptr += 1
                cur_start, cur_end, cur_label = entity_spans[entity_ptr]
        else:
            tokens_and_labels.append([token, "O"])
    tokens_and_labels = maybe_typos(tokens_and_labels, prob=0.15)
    return {"text": " ".join([tok for tok, _ in tokens_and_labels]), "bio_labels": tokens_and_labels}

def main():
    dataset = [generate_sample() for _ in range(TOTAL_SAMPLES)]
    with open(DEST_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Сгенерировано {TOTAL_SAMPLES} записей и сохранено в {DEST_PATH}")

if __name__ == "__main__":
    main()
