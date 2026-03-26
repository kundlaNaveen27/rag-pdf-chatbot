# Version 4 — Multiple PDF support
# Improvement over v3: load and search across multiple PDFs simultaneously

import os
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# STEP 1: Extract text from PDF — same as before
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


# STEP 2: Split into chunks — same as v2/v3
def split_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_text(text)


# STEP 3: Create embeddings — same as before
def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return embeddings, model


# STEP 4: Build FAISS index from ALL chunks across ALL PDFs
# NEW — takes combined chunks and builds one big index
def build_index(all_chunks, embedding_model):
    embeddings = embedding_model.encode(all_chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings


# STEP 5: Search relevant chunks — same as before
def search_relevant_chunks(question, embedding_model, index, all_chunks, top_k=3):
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    relevant_chunks = [all_chunks[i] for i in indices[0]]
    chunk_numbers = [int(i) for i in indices[0]]
    return relevant_chunks, chunk_numbers


# STEP 6: Answer question — same as before
def answer_question(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful research assistant analyzing academic papers.
                Answer questions based on the provided context chunks.
                If multiple papers are referenced, compare and contrast them.
                If the exact answer isn't in the context, use what IS there to give
                the best possible answer. Only say 'I could not find that' if the
                context has absolutely no relevant information."""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content


# ── MAIN PROGRAM ─────────────────────────────────────

print("\n📚 RAG Chatbot — Chat with Multiple PDFs\n")
print("Commands:")
print("  load <filename>  → load a PDF")
print("  list             → see all loaded PDFs")
print("  quit             → exit\n")

# Load embedding model once — reuse for all PDFs
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready!\n")

# NEW — dictionary to store info about each loaded PDF
# key = filename, value = list of chunks from that PDF
# {"sample.pdf": ["chunk1...", "chunk2..."], "paper2.pdf": [...]}
loaded_pdfs = {}

# Combined list of ALL chunks from ALL PDFs
all_chunks = []

# FAISS index — starts empty, rebuilt each time a PDF is loaded
index = None

# CHAT LOOP
while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    # ── COMMAND: list ──────────────────────────────
    elif user_input.lower() == "list":
        if not loaded_pdfs:
            print("No PDFs loaded yet. Use: load <filename>\n")
        else:
            print("\n📄 Loaded PDFs:")
            for filename, chunks in loaded_pdfs.items():
                print(f"  {filename} — {len(chunks)} chunks")
            print()

    # ── COMMAND: load <filename> ───────────────────
    elif user_input.lower().startswith("load "):
        # extract filename from command
        # "load sample.pdf" → "sample.pdf"
        filename = user_input[5:].strip()

        if not os.path.exists(filename):
            print(f"❌ File '{filename}' not found in current folder\n")

        elif filename in loaded_pdfs:
            print(f"⚠️ '{filename}' already loaded\n")

        else:
            print(f"Loading {filename}...")
            text = extract_text_from_pdf(filename)
            chunks = split_into_chunks(text)

            # store this PDF's chunks in dictionary
            loaded_pdfs[filename] = chunks

            # add to combined list
            all_chunks.extend(chunks)  # extend adds all items from a list

            # rebuild FAISS index with all chunks including new ones
            print("Rebuilding search index...")
            index, _ = build_index(all_chunks, embedding_model)

            print(f"✅ Loaded '{filename}' — {len(chunks)} chunks")
            print(f"   Total chunks across all PDFs: {len(all_chunks)}\n")

    # ── QUESTION: answer from loaded PDFs ─────────
    else:
        if index is None:
            print("⚠️ No PDFs loaded yet. Use: load <filename>\n")
        else:
            relevant_chunks, chunk_numbers = search_relevant_chunks(
                user_input, embedding_model, index, all_chunks
            )
            answer = answer_question(user_input, relevant_chunks)
            print(f"\nBot: {answer}\n")

            print("📍 Sources:")
            for num, chunk in zip(chunk_numbers, relevant_chunks):
                preview = chunk[:150].replace("\n", " ")
                print(f"  Chunk {num}: \"{preview}...\"")
            print("-" * 50 + "\n")