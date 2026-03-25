# RAG PDF Chatbot 🤖

Ask questions about any PDF document and get AI-powered answers 
grounded in your document — not hallucinated.

## What This Does
Upload any PDF and chat with it. The app finds the most relevant 
sections and answers your question based only on those sections.

## How It Works
1. PDF is loaded and split into smart chunks (LangChain)
2. Chunks converted to embeddings (SentenceTransformers)
3. Embeddings stored in FAISS vector database
4. Your question is matched to relevant chunks semantically
5. LLaMA 3.3 70B answers based only on relevant chunks

## Tech Stack
- LangChain — smart sentence-aware text splitting
- FAISS — vector similarity search
- SentenceTransformers (all-MiniLM-L6-v2) — text embeddings
- Groq + LLaMA 3.3 70B — AI response generation
- PyPDF2 — PDF text extraction

## Setup
bash
pip install langchain-text-splitters sentence-transformers faiss-cpu pypdf2 groq python-dotenv


Create a .env file:
GROQ_API_KEY=your_key_here

Run:
bash
python rag_chatbot.py


## Example
You: What problem does this paper solve?
Bot: The paper addresses the limitation of sequential computation 
in RNNs by proposing the Transformer architecture based entirely 
on attention mechanisms...

## What I Learned
- RAG pipeline architecture from scratch
- Vector embeddings and semantic search
- FAISS indexing and similarity search
- LangChain text splitting strategies
- Prompt engineering for grounded responses
