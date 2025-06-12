import json
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE = Path(__file__).parent.parent.parent
DATA_PATH = BASE / "classifier" / "data" / "router_balanced_data.json"
MODEL_PATH = BASE / "classifier" / "models" / "router_clf.joblib"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)
X = [d["query"] for d in data]
y = [d["label"] for d in data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

clf = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
    ("lr", LogisticRegression(max_iter=300))
])
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test), digits=3))
joblib.dump(clf, MODEL_PATH)
print(f"✔ Модель сохранена: {MODEL_PATH}")
