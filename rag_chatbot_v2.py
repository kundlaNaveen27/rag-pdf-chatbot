import os
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── STEP 1: Extract text from PDF ──────────────────
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        print(f"PDF has {len(reader.pages)} pages")
        for page in reader.pages:
            text += page.extract_text()
    return text


# ── STEP 2: Split text into chunks ─────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # max characters per chunk
        chunk_overlap=50,     # 50 characters shared between chunks
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(text)
    return chunks


# ── STEP 3: Convert chunks to embeddings ───────────
def create_embeddings(chunks):
    print("Creating embeddings... (this takes ~30 seconds first time)")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return embeddings, model


# ── STEP 4: Store embeddings in FAISS ──────────────
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]   # size of each embedding vector
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))   # add all embeddings to index
    return index


# ── STEP 5: Search for relevant chunks ─────────────
def search_relevant_chunks(question, model, index, chunks, top_k=3):
    # convert question to embedding
    question_embedding = model.encode([question])

    # search FAISS for top_k most similar chunks
    distances, indices = index.search(np.array(question_embedding), top_k)

    # return the actual text of those chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks


# ── STEP 6: Answer question using relevant chunks ──
def answer_question(question, relevant_chunks):
    # join chunks into one context block
    context = "\n\n".join(relevant_chunks)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant that answers questions 
                based ONLY on the provided context. If the answer is not in the 
                context, say 'I could not find that in the document.'"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content


# ── MAIN PROGRAM ────────────────────────────────────

print("\n📚 RAG Chatbot — Chat with your PDF\n")

# Build the index once
print("Loading PDF...")
text = extract_text_from_pdf("sample.pdf")

print("Splitting into chunks...")
chunks = split_into_chunks(text)
print(f"Created {len(chunks)} chunks")

embeddings, embedding_model = create_embeddings(chunks)
print(f"Created {len(embeddings)} embeddings")

index = create_faiss_index(embeddings)
print("FAISS index ready!\n")

print("=" * 50)
print("You can now ask questions about your PDF!")
print("Type 'quit' to exit\n")

# Chat loop
while True:
    question = input("You: ")

    if question.lower() == "quit":
        print("Goodbye!")
        break

    relevant_chunks = search_relevant_chunks(
        question, embedding_model, index, chunks
    )

    answer = answer_question(question, relevant_chunks)
    print(f"\nBot: {answer}\n")
    print("-" * 50)