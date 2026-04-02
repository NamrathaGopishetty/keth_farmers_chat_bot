from pdf_utils import load_pdfs, split_documents
from rag import create_vector_db

print("📄 Loading PDFs...")
docs = load_pdfs()

print("✂️ Splitting...")
chunks = split_documents(docs)

print("🧠 Creating FAISS...")
create_vector_db(chunks)

print("✅ DONE!")