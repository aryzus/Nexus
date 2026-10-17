import os
import hashlib
import json
import numpy as np
from langchain_core.documents import Document
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_simple_embedding(text):
    """Simple TF-IDF style embedding using character n-grams"""
    text = text.lower()[:500]
    vocab_size = 1536
    embedding = np.zeros(vocab_size)
    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        idx = int(hashlib.md5(trigram.encode()).hexdigest(), 16) % vocab_size
        embedding[idx] += 1
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []

    def add_documents(self, chunks, metadata):
        print(f"Embedding {len(chunks)} chunks...")
        for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
            emb = get_simple_embedding(chunk)
            self.documents.append(chunk)
            self.embeddings.append(emb)
            self.metadata.append(meta)
            if (i + 1) % 500 == 0:
                print(f"  Embedded {i+1}/{len(chunks)} chunks")
        print("Vector store ready!")

    def similarity_search(self, query, k=8):
        query_emb = get_simple_embedding(query)
        similarities = [cosine_similarity(query_emb, emb) for emb in self.embeddings]
        top_indices = np.argsort(similarities)[-k:][::-1]
        results = []
        for idx in top_indices:
            results.append({
                'content': self.documents[idx],
                'metadata': self.metadata[idx],
                'score': similarities[idx]
            })
        return results

def build_vector_store(chunks, metadata):
    print("Building vector store...")
    store = SimpleVectorStore()
    store.add_documents(chunks, metadata)
    return store

def retrieve_context(vector_store, query, k=8):
    docs = vector_store.similarity_search(query, k=k)
    context_parts = []
    sources = []
    for doc in docs:
        source = doc['metadata'].get('source', 'Unknown')
        page = doc['metadata'].get('page', '?')
        context_parts.append(f"[Source: {source}, Page {page}]\n{doc['content']}")
        sources.append(f"{source} (p.{page})")
    return "\n\n---\n\n".join(context_parts), list(set(sources))

def ask_nexus(vector_store, question, doc_names, chat_history=None):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    context, sources = retrieve_context(vector_store, question)

    history_text = ""
    if chat_history:
        for msg in chat_history[-4:]:
            role = "User" if msg["role"] == "user" else "Nexus"
            history_text += f"{role}: {msg['content']}\n"

    system_prompt = f"""You are Nexus — an intelligent multi-document AI assistant.
You have access to: {', '.join(doc_names)}

RULES:
1. Only answer based on the provided document context
2. Always cite which document and page number your answer comes from
3. Compare documents when asked — provide specific differences
4. If information is not in the documents say so clearly
5. Be specific — use exact figures, dates and names from the documents
6. Never make up information not in the context
"""

    user_message = f"""Document Context:
{context}

{f"Previous conversation:{chr(10)}{history_text}" if history_text else ""}

Question: {question}

Answer based strictly on the document context. Cite sources."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1,
        max_tokens=1000,
    )

    return response.choices[0].message.content.strip(), sources

def generate_document_summary(vector_store, doc_name):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    context, _ = retrieve_context(
        vector_store,
        f"executive summary overview key highlights {doc_name}",
        k=10
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are an expert document analyst."},
            {"role": "user", "content": f"""Based on this content from {doc_name}:

{context}

Generate a structured summary with:
1. Company overview (2-3 sentences)
2. Key highlights (5 bullet points)
3. Important numbers and metrics
4. Main themes covered

Use actual data from the document."""}
        ],
        temperature=0.2,
        max_tokens=800,
    )

    return response.choices[0].message.content.strip()