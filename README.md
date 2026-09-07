# RAG-Based Chatbot — Indian Classical Music

A simple student-friendly Retrieval-Augmented Generation (RAG) chatbot.

## Tech used
- Python
- Streamlit
- FAISS
- NumPy
- OpenAI API
- PyPDF
- python-dotenv

## Important
This version intentionally does NOT use `sentence-transformers` or `torch`.
That makes installation much easier on Windows and avoids the long-path
`WinError 206` problem encountered with PyTorch.

## Run

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If you use an OpenAI API key, create `.env`:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

Never upload `.env` to GitHub.

## RAG pipeline

Documents → Chunking → Vector representation → FAISS retrieval → LLM → Answer

The included demo knowledge base is about Indian Classical Music. Replace
the files in `data/` to adapt the chatbot to another niche domain.
