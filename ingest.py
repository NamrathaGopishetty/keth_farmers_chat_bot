# ingest.py
import os
import re
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag import create_vector_db

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text += page_text + "\n"
    return text

def is_garbage_chunk(chunk):
    if len(chunk.strip()) < 60:
        return True

    # ✅ Filter Tamil and non-Latin scripts
    tamil_chars = sum(1 for c in chunk if '\u0B80' <= c <= '\u0BFF')
    if tamil_chars > 5:
        return True

    # ✅ Filter chunks that are mostly non-English
    english_chars = sum(1 for c in chunk if c.isascii())
    if (english_chars / len(chunk)) < 0.75:
        return True

    # ✅ Filter OCR garbage
    special_chars = sum(1 for c in chunk if c in '~@#$%^&*[]{}|<>\\/')
    if (special_chars / len(chunk)) > 0.05:
        return True

    # ✅ Filter TOC lines (mostly page numbers)
    lines = chunk.strip().split("\n")
    page_number_lines = sum(1 for l in lines if re.search(r'\b\d{1,4}\s*$', l.strip()))
    if len(lines) > 0 and (page_number_lines / len(lines)) > 0.4:
        return True

    # ✅ Filter mostly numeric/symbol chunks
    non_alpha = sum(1 for c in chunk if not c.isalpha())
    if (non_alpha / len(chunk)) > 0.6:
        return True

    # ✅ Filter known headers
    lower = chunk.lower()
    skip_phrases = ["s.no", "page no", "table of contents", "contents"]
    if any(phrase in lower for phrase in skip_phrases):
        return True

    return False

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    clean = [c.strip() for c in chunks if not is_garbage_chunk(c)]
    return clean

def ingest_all(data_folder="data"):
    if not os.path.exists(data_folder):
        print(f"❌ Folder '{data_folder}' not found!")
        return

    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"❌ No PDFs found in '{data_folder}'!")
        return

    print(f"📂 Found {len(pdf_files)} PDF(s)\n")
    all_chunks = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        print(f"📄 Processing: {pdf_file}")
        text = extract_text_from_pdf(pdf_path)

        if not text.strip():
            print(f"   ❌ No text extracted — skipping\n")
            continue

        chunks = chunk_text(text)
        print(f"   ✅ {len(chunks)} clean English chunks")

        # Show 2 sample chunks per PDF
        for i, chunk in enumerate(chunks[:2]):
            print(f"   Sample {i+1}: {chunk[:150]}\n")

        all_chunks.extend(chunks)

    if not all_chunks:
        print("❌ No chunks created!")
        return

    print(f"\n📊 Total clean chunks: {len(all_chunks)}")
    create_vector_db(all_chunks)
    print("✅ Vector DB ready!")

if __name__ == "__main__":
    ingest_all("data")