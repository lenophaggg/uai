from fastapi import FastAPI, UploadFile, File
from elasticsearch import Elasticsearch
import pdfplumber
import docx
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

app = FastAPI()

es = Elasticsearch("http://localhost:9200")
INDEX_NAME = "documents"

def extract_pdf_text(file):
    file.seek(0)
    with pdfplumber.open(file) as pdf:
        text = "\n".join([p.extract_text() or "" for p in pdf.pages])
        if text.strip():
            return text

    # OCR fallback
    print("PDF пустой — включаем OCR")
    file.seek(0)
    images = convert_from_bytes(file.read())
    return "\n".join(
        pytesseract.image_to_string(img, lang='rus+eng') for img in images
    )

def extract_docx_text(file):
    file.seek(0)
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs)

def split_with_overlap(text, chunk_size=500, step=400, min_length=30):
    fragments = []
    length = len(text)
    for i in range(0, length, step):
        frag = text[i:i+chunk_size].strip()
        if len(frag) >= min_length:
            fragments.append(frag)
        if i + chunk_size >= length:
            tail = text[i+step:].strip()
            if len(tail) >= min_length and tail not in fragments:
                fragments.append(tail)
            break
    return fragments

@app.post("/upload_doc/")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename
    if filename.endswith('.pdf'):
        content = extract_pdf_text(file.file)
    elif filename.endswith('.docx'):
        content = extract_docx_text(file.file)
    else:
        return {"error": "only .pdf and .docx supported"}

    # 1. Пробуем разбить по абзацам (по двойным переносам)
    fragments = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 30]
    # 2. Если абзацев мало (0 или 1), режем на куски с overlap
    if len(fragments) < 2:
        fragments = split_with_overlap(content, chunk_size=500, step=400, min_length=30)

    for i, fragment in enumerate(fragments):
        doc = {
            "filename": filename,
            "fragment_num": i + 1,
            "content": fragment,
        }
        es.index(index=INDEX_NAME, document=doc)
    return {"status": "uploaded", "filename": filename, "fragments": len(fragments)}
