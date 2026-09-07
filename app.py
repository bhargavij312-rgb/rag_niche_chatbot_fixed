import os
import re
from pathlib import Path
from collections import Counter

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import numpy as np
import faiss
from openai import OpenAI

load_dotenv()

DATA_DIR = Path("data")

def load_documents():
    texts = []
    for path in DATA_DIR.glob("*.txt"):
        texts.append(path.read_text(encoding="utf-8"))
    for path in DATA_DIR.glob("*.pdf"):
        reader = PdfReader(str(path))
        texts.append("\n".join(page.extract_text() or "" for page in reader.pages))
    return texts

def chunk_text(text, size=120, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += max(1, size - overlap)
    return chunks

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def make_vectors(chunks):
    # Lightweight TF-IDF-like vectors using only Python + NumPy.
    # This avoids sentence-transformers/PyTorch installation problems on Windows.
    vocabulary = {}
    for chunk in chunks:
        for word in set(tokenize(chunk)):
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)

    matrix = np.zeros((len(chunks), len(vocabulary)), dtype="float32")
    doc_freq = np.zeros(len(vocabulary), dtype="float32")

    for i, chunk in enumerate(chunks):
        counts = Counter(tokenize(chunk))
        for word in counts:
            idx = vocabulary[word]
            matrix[i, idx] = 1 + np.log(counts[word])
            doc_freq[idx] += 1

    idf = np.log((1 + len(chunks)) / (1 + doc_freq)) + 1
    matrix *= idf

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    return matrix, vocabulary, idf

@st.cache_resource
def build_index():
    docs = load_documents()
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))

    if not chunks:
        return None, [], None, None

    vectors, vocabulary, idf = make_vectors(chunks)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, chunks, vocabulary, idf

def retrieve(question, index, chunks, vocabulary, idf, k=4):
    q_counts = Counter(tokenize(question))
    q = np.zeros((1, len(vocabulary)), dtype="float32")
    for word, count in q_counts.items():
        if word in vocabulary:
            q[0, vocabulary[word]] = (1 + np.log(count)) * idf[vocabulary[word]]

    norm = np.linalg.norm(q)
    if norm > 0:
        q /= norm

    scores, ids = index.search(q, min(k, len(chunks)))
    results = [(float(scores[0][j]), chunks[i]) for j, i in enumerate(ids[0]) if i >= 0]
    return results

def generate_answer(question, context):
    if not context.strip():
        return (
            "🤔 Hmm, I couldn't find that in my notes yet. "
            "Try asking about a raga, tala, or instrument I know about!"
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "OPENAI_API_KEY is not configured, so here are the most relevant "
            "retrieved passages from the knowledge base:\n\n" + context
        )

    prompt = f"""You are a helpful chatbot specialized in Indian Classical Music.
Answer using ONLY the supplied context. If the context does not contain enough
information, say that you do not have enough information. Keep the answer
student-friendly and concise.

Context:
{context}

Question: {question}
"""
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "Answer only from the retrieved context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception:
        return (
            "😅 I couldn't reach the AI service right now (this usually means the API "
            "key has no billing credit, or there's a network issue).\n\n"
            "Here's what I found directly in my notes instead:\n\n"
            + context
        )
    
st.set_page_config(page_title="RAG Music Chatbot", page_icon="🎵")
st.title("🎵 RAG Chatbot — Indian Classical Music")
st.caption("Retrieval-Augmented Generation demo using a niche knowledge base.")

DATA_DIR.mkdir(exist_ok=True)
index, chunks, vocabulary, idf = build_index()

if index is None:
    st.warning("No documents found in the data/ folder.")
else:
    question = st.chat_input("Ask about ragas, talas, gharanas, instruments, etc.")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant information..."):
                results = retrieve(question, index, chunks, vocabulary, idf)
                context = "\n\n---\n\n".join(text for _, text in results)
                st.write(generate_answer(question, context))

            with st.expander("Retrieved context"):
                for score, text in results:
                    st.write(f"Similarity: {score:.3f}")
                    st.write(text)
                    st.write("---")

with st.sidebar:
    st.header("How RAG works")
    st.write("1. Load documents")
    st.write("2. Split into chunks")
    st.write("3. Create text vectors")
    st.write("4. FAISS similarity search")
    st.write("5. Send retrieved context to the LLM")
