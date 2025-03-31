import os
import fitz  # PyMuPDF
import numpy as np
import faiss
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --------------- CONFIGURATION ---------------- #
PDF_FOLDER = "PDF"  # Folder where PDFs are stored
DB_PATH = "vector_database"  # Folder to store FAISS index & metadata
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Change if needed
CHUNK_SIZE = 500  # Optimized for retrieval
CHUNK_OVERLAP = 50  # Small overlap for better context
USE_GPU = False  # Change to True if using faiss-gpu

# --------------- SETUP ---------------- #
os.makedirs(DB_PATH, exist_ok=True)

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# Function to process PDFs and create chunks
def process_pdfs(pdf_folder):
    all_chunks = []
    metadata = []
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            book_text = extract_text_from_pdf(pdf_path)
            chunks = splitter.split_text(book_text)
            
            for chunk in chunks:
                all_chunks.append(chunk)
                metadata.append({"book": filename})  # Store book name for filtering

    return all_chunks, metadata

# Function to create FAISS index and store it
def create_faiss_index(text_chunks, metadata, save_path):
    print("Generating embeddings...")
    embeddings = model.encode(text_chunks, batch_size=32, show_progress_bar=True)
    
    # Convert to NumPy array
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Exact search
    index.add(embeddings)  # Add embeddings to index

    # Move to GPU if enabled
    if USE_GPU:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Save FAISS index
    faiss.write_index(index, os.path.join(save_path, "faiss_index"))

    # Save metadata
    with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print("FAISS index & metadata saved successfully!")

# Function to load FAISS index & metadata
def load_faiss_index(load_path):
    index = faiss.read_index(os.path.join(load_path, "faiss_index"))

    with open(os.path.join(load_path, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

# ----------------- EXECUTION ----------------- #
if __name__ == "__main__":
    print("Processing PDFs...")
    text_chunks, metadata = process_pdfs(PDF_FOLDER)
    
    print(f"Total chunks created: {len(text_chunks)}")
    
    create_faiss_index(text_chunks, metadata, DB_PATH)
