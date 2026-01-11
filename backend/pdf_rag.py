import fitz  # PyMuPDF
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import requests

# Global variables for embedding model only
_EMBEDDING_MODEL = None

def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        print("Loading embedding model...")
        _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBEDDING_MODEL

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def create_chunks(text: str, chunk_size: int = 500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def process_pdf(pdf_bytes: bytes):
    # 1. Extract Text
    text = extract_text_from_pdf(pdf_bytes)
    
    # 2. Chunk Text
    chunks = create_chunks(text)
    if not chunks:
        return {"error": "No text found in PDF"}
    
    # 3. Embed Chunks
    model = get_embedding_model()
    embeddings = model.encode(chunks)
    
    # 4. Create FAISS Index
    dimension = embeddings.shape[1]
    vector_index = faiss.IndexFlatL2(dimension)
    vector_index.add(np.array(embeddings))
    
    print(f"Processed PDF. Created {len(chunks)} chunks.")
    return {"message": "PDF processed successfully", "chunks_count": len(chunks), "vector_index": vector_index, "chunks": chunks}

def ask_pdf_with_data(question: str, api_key: str, vector_index, chunks):
    # 1. Embed Question
    model = get_embedding_model()
    question_embedding = model.encode([question])
    
    # 2. Search FAISS
    k = 3
    D, I = vector_index.search(np.array(question_embedding), k)
    
    # 3. Retrieve Context
    context = ""
    for idx in I[0]:
        if idx < len(chunks):
            context += chunks[idx] + "\n\n"
            
    # 4. Ask Groq
    if not api_key:
        return "API Key not set."

    system_prompt = f"You are a helpful assistant. Answer the user question based strictly on the context provided below.\n\nContext:\n{context}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "temperature": 0.5
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error querying Groq: {str(e)}"