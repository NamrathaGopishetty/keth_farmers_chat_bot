# 🌾 Farmer Assistant AI

An intelligent AI-powered assistant designed to help farmers make better decisions using **Retrieval-Augmented Generation (RAG)**, **Natural Language Processing (NLP)**, and an interactive **chat + voice interface**.

---

## 🚀 Features

* 📄 **PDF Knowledge Base**
  Upload agricultural documents (crop guides, soil data, weather info) and extract insights.

* 🧠 **RAG (Retrieval-Augmented Generation)**
  Combines FAISS-based document retrieval with AI generation for accurate answers.

* 💬 **Chat Interface**
  User-friendly conversational UI built using Streamlit.

* 🎤 **Voice Assistant**
  Ask questions using voice input (speech-to-text integration).

* ⚡ **Fast Semantic Search**
  Uses Sentence Transformers for efficient similarity search.

* 🌱 **Real-World Impact**
  Helps farmers with crop selection, irrigation advice, and best practices.

---

## 🛠️ Tech Stack

* **Frontend/UI**: Streamlit
* **Backend**: Python
* **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector Database**: FAISS
* **LLM**: HuggingFace FLAN-T5
* **Voice Processing**: Web Speech API / SpeechRecognition

---

## 📂 Project Structure

```
Farmer-Assistant/
│
├── app.py              # Streamlit UI
├── rag.py              # RAG pipeline (search + generation)
├── pdf_utils.py        # PDF loading & splitting
├── setup_db.py         # Create FAISS index
├── faiss_index.index   # Vector database
├── chunks.pkl          # Stored text chunks
└── requirements.txt
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/farmer-assistant.git
cd farmer-assistant
```

### 2️⃣ Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## 📄 Add Your PDFs

Place all agricultural PDFs inside a folder (e.g., `data/`)

---

## 🧠 Create Vector Database

```
python setup_db.py
```

This will:

* Load PDFs
* Split into chunks
* Generate embeddings
* Store in FAISS

---

## ▶️ Run the App

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 💡 Example Questions

* “Which crop is suitable in winter?”
* “How to improve soil fertility?”
* “Best irrigation methods for cotton?”

---

## 📈 How It Works

1. User asks a question (text or voice)
2. Query is converted into embeddings
3. FAISS retrieves relevant document chunks
4. Context is passed to LLM (FLAN-T5)
5. AI generates a precise answer

---

## 🎯 Future Improvements

* 🌍 Multi-language support (Telugu, Hindi)
* 📊 Weather API integration
* 📱 Mobile-friendly UI
* 🤖 More powerful LLM integration
* 🛰️ Satellite data for crop monitoring

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork and improve the project.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 🙌 Acknowledgements

* HuggingFace Transformers
* Sentence Transformers
* FAISS
* Streamlit

---

## 👩‍💻 Author

**Namratha Gopishetty**

---

⭐ If you found this useful, give it a star!
