# rag.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from groq import Groq
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- EMBEDDING MODEL ----------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- GROQ CLIENT ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")  # 👈 or paste key directly
GROQ_MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY)

_index = None
_chunks = None

# ---------------- CREATE VECTOR DB ----------------
def create_vector_db(chunks):
    if not chunks:
        raise ValueError("No chunks provided.")

    print(f"🔄 Encoding {len(chunks)} chunks...")
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))

    faiss.write_index(index, "faiss_index.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Vector DB created with {len(chunks)} chunks.")


# ---------------- LOAD VECTOR DB ----------------
def load_vector_db():
    global _index, _chunks

    if _index is not None and _chunks is not None:
        return _index, _chunks

    if not os.path.exists("faiss_index.index") or not os.path.exists("chunks.pkl"):
        raise FileNotFoundError("Vector DB not found. Run ingest.py first.")

    _index = faiss.read_index("faiss_index.index")
    with open("chunks.pkl", "rb") as f:
        _chunks = pickle.load(f)

    print(f"✅ Vector DB loaded: {len(_chunks)} chunks.")
    return _index, _chunks


# ---------------- PREPARE CONTEXT ----------------
def prepare_context(results):
    seen = set()
    clean_chunks = []
    for chunk in results:
        chunk = " ".join(chunk.split())
        if chunk not in seen and len(chunk) >= 60:
            seen.add(chunk)
            clean_chunks.append(chunk)
    return "\n\n".join(clean_chunks)


# ---------------- SEARCH ----------------
def search(query, top_k=5):
    try:
        index, chunks = load_vector_db()
    except FileNotFoundError as e:
        print(f"Search error: {e}")
        return []

    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(
        np.array(query_embedding, dtype=np.float32), top_k
    )

    results = [chunks[i] for i in indices[0] if i != -1]
    return results


# ---------------- BUILD PROMPT ----------------
def build_prompt(query, context):
    return f"""You are an expert agricultural assistant helping farmers in India.

You have been provided some context from a knowledge base below.
Use the context as your primary source. If the context doesn't have enough
information, use your own agricultural knowledge to give a helpful answer.

Always:
- Give a clear, complete, and detailed answer
- Use simple language a farmer can understand
- Number the steps if steps are involved
- Include quantities and measurements wherever possible
- Never say "I don't have information" — always try to help

Context from knowledge base:
{context}

Question: {query}"""


# ---------------- GENERATE ANSWER ----------------
def generate_answer(query, chat_history=None):
    results = search(query)

    if not results:
        context = "No specific context found in knowledge base."
    else:
        context = prepare_context(results)

    prompt = build_prompt(query, context)

    # ✅ Build message history for Groq
    messages = [
        {
            "role": "system",
            "content": """You are an expert agricultural assistant for Indian farmers
with deep knowledge of crops, fertilizers, irrigation, pest control, and soil management.
Always give helpful, detailed, practical answers even if the context is limited.
Never refuse to answer — use your own knowledge to fill any gaps.
Remember the conversation history and refer to it when relevant.
Format your answers clearly with numbered steps where applicable."""
        }
    ]

    # ✅ Add previous chat history so Groq remembers context
    if chat_history:
        for msg in chat_history[-6:]:   # ✅ Last 6 messages to avoid token limit
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    # ✅ Add current question with context
    messages.append({
        "role": "user",
        "content": prompt
    })

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=600,
        )

        answer = response.choices[0].message.content.strip()
        return answer if answer else "Sorry, I could not generate an answer. Please try again."

    except Exception as e:
        return f"⚠️ Groq API error: {e}"
