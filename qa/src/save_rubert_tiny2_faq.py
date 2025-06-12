from sentence_transformers import SentenceTransformer

# Загружаем модель с Hugging Face
model = SentenceTransformer("cointegrated/rubert-tiny2")

# Сохраняем модель в нужную папку
model.save(r"C:\Users\renzh\OneDrive\Desktop\uai\qa\models\rubert-tiny2-faq")

print("Модель сохранена корректно!")
