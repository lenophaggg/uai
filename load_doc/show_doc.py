from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "documents"

def show_document_content(filename):
    # Ищем все фрагменты по filename, сортируем по fragment_num если есть
    query = {
        "query": {
            "term": {
                "filename.keyword": filename
            }
        },
        "sort": [
            {"fragment_num": {"order": "asc"}}
        ]
    }
    search = es.search(index=INDEX_NAME, body=query, size=1000)  # увеличить size для больших доков
    hits = search["hits"]["hits"]
    if not hits:
        print(f"Документ с именем {filename} не найден!")
        return

    print(f"Содержимое документа {filename}:")
    print("-" * 40)
    for hit in hits:
        fragment_num = hit["_source"].get("fragment_num", None)
        content = hit["_source"].get("content", "")
        prefix = f"[Фрагмент {fragment_num}]" if fragment_num is not None else ""
        print(f"{prefix}\n{content}\n")
    print("-" * 40)

if __name__ == "__main__":
    filename = input("Введите имя файла для просмотра: ").strip()
    show_document_content(filename)
