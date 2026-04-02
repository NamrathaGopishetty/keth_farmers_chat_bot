from pypdf import PdfReader
import os

def load_pdfs(folder_path="data"):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            text = ""

            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content

            documents.append(text)

    return documents


def split_documents(documents, chunk_size=300):
    chunks = []

    for doc in documents:
        for i in range(0, len(doc), chunk_size):
            chunks.append(doc[i:i+chunk_size])

    return chunks