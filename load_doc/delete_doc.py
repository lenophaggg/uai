from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "documents"

def delete_documents_by_filename(filename):
    query = {
        "query": {
            "term": {
                "filename.keyword": filename
            }
        }
    }
    resp = es.delete_by_query(index=INDEX_NAME, body=query)
    print(f"Удалено документов: {resp['deleted']}")

if __name__ == "__main__":
    filename = input("Введите имя файла для удаления: ").strip()
    delete_documents_by_filename(filename)
