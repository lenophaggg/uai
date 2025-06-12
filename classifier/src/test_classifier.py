import joblib
from pathlib import Path

# Настрой путь к модели
BASE = Path(__file__).parent
MODEL_PATH = r"C:\Users\renzh\OneDrive\Desktop\uai\classifier\models\router_clf.joblib"
clf = joblib.load(MODEL_PATH)

def main():
    print("Введите запрос (пустая строка — выход):")
    while True:
        query = input("> ").strip()
        if not query:
            break
        label = clf.predict([query])[0]
        print(f"Тип запроса: {label}")

if __name__ == "__main__":
    main()
