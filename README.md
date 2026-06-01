# ◈ DocMind — AI Document Chat

> Upload any PDF and have a real conversation with it. Powered by Mistral AI and RAG.

---

## What is DocMind?

DocMind is a production-grade RAG (Retrieval-Augmented Generation) application built with Streamlit. Upload a PDF, and the app embeds it into an in-memory vector database. You can then ask questions about the document in a chat interface — the AI retrieves the most relevant chunks and answers strictly from the document content. It remembers conversation context, so follow-up questions work naturally.

---

## Features

- **Chat with any PDF** — ask questions in plain English
- **Conversation memory** — understands follow-up questions like "explain it simpler"
- **Confidence score** — every answer shows how confident the retrieval was
- **Source pages** — see exactly which pages the answer came from
- **No hallucination** — AI only answers from the document, never outside knowledge
- **Large PDF support** — handles up to 300 pages per session
- **Premium dark UI** — production-quality interface, not a default Streamlit look

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI + Server | Streamlit |
| LLM | Mistral AI (`mistral-small-2603`) |
| Embeddings | Mistral AI (`mistral-embed`) |
| Vector DB | ChromaDB (in-memory) |
| PDF Loader | LangChain PyPDFLoader |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |

---

## Project Structure

```
RAG/
├── app.py              # Full application (frontend + backend)
├── requirements.txt    # Python dependencies
├── .env                # API keys (never commit this)
├── .gitignore
└── README.md
```

---

## Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/docmind.git
cd docmind
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Create a `.env` file**
```
MISTRAL_API_KEY=your_mistral_api_key_here
```
Get your key at [console.mistral.ai](https://console.mistral.ai)

**4. Run the app**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How to Use

1. **Upload** — click the "Upload Document" panel and select a PDF
2. **Index** — click "Index Document" and wait for embedding to finish
3. **Ask** — type any question about the document and press Enter or Send

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo, set main file as `app.py`
4. Add your secret under **Advanced Settings**:
```toml
MISTRAL_API_KEY = "your_key_here"
```
5. Click Deploy — your app gets a public URL instantly

---

## Environment Variables

| Variable | Description |
|---|---|
| `MISTRAL_API_KEY` | Your Mistral AI API key |

---

## Notes

- The vector store lives in RAM — it resets when the session ends or the server restarts
- For PDFs over 300 pages, only the first 300 pages are indexed
- The `chroma_db/` folder in your project (if present) is a separate local index from `create_database.py` — the Streamlit app never touches it

---

## License

MIT
