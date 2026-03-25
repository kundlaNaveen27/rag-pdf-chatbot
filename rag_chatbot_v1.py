import os
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# STEP 1: Read all text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:  # "rb" = read binary (PDFs are encoded files)
        reader = PyPDF2.PdfReader(file)
        print(f"PDF has {len(reader.pages)} pages")
        for page in reader.pages:
            text += page.extract_text()  # += adds each page to the bucket
    return text


# STEP 2: Cut big text into 500-word pieces
# Why? AI can't receive 50,000 words at once — only send relevant chunks
def split_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        current_chunk.append(word)
        current_size += 1

        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))  # join words back into string
            current_chunk = []
            current_size = 0

    if current_chunk:  # save leftover words as final chunk
        chunks.append(" ".join(current_chunk))

    return chunks


# STEP 3: Convert chunks to numbers (embeddings)
# Similar meaning = similar numbers — this enables semantic search
def create_embeddings(chunks):
    print("Creating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # converts text → 384 numbers
    embeddings = model.encode(chunks)
    return embeddings, model  # return model too — needed later for questions


# STEP 4: Store embeddings in FAISS (vector database)
# Like building a library catalog — add once, search anytime instantly
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # how many numbers per embedding = 384
    index = faiss.IndexFlatL2(dimension)  # L2 = straight-line distance formula
    index.add(np.array(embeddings))
    return index


# STEP 5: Find 3 most relevant chunks for the question
# Converts question to numbers → FAISS finds chunks with similar numbers
def search_relevant_chunks(question, model, index, chunks, top_k=3):
    question_embedding = model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]  # grab text at matching positions
    return relevant_chunks


# STEP 6: Send relevant chunks + question to AI
# System prompt forces AI to answer ONLY from context — prevents hallucination
def answer_question(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """Answer questions based ONLY on the provided context. 
                If answer not found say 'I could not find that in the document.'"""
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

# SETUP — runs once to build searchable index
text = extract_text_from_pdf("sample.pdf")
chunks = split_into_chunks(text)
print(f"Created {len(chunks)} chunks")

embeddings, embedding_model = create_embeddings(chunks)
index = create_faiss_index(embeddings)
print("FAISS index ready!\n")

print("You can now ask questions about your PDF!")
print("Type 'quit' to exit\n")

# CHAT — runs for every question
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