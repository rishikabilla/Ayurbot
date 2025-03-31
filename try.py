import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Loading the vector database
KB_path = "vector_database"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
kb = FAISS.load_local(KB_path, embedding_model, allow_dangerous_deserialization=True)
retriever = kb.as_retriever(search_kwargs={"k": 3})  # Fetch only top 1 chunk for speed

# Initialize optimized LLM
llm = OllamaLLM(model="mistral", num_predict=100,max_tokens=150)


# Create a RAG pipeline
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

query = input("Ask Ayurbot: ")

# Step 1: Start timer
start_time = time.time()

# Step 2: FAISS Retrieval
search_start = time.time()
retrieved_docs = retriever.invoke(query)
search_end = time.time()
print(f"FAISS retrieval time: {search_end - search_start:.2f} seconds")

# Step 3: Pre-process text to avoid overloading LLM
retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_chunks = text_splitter.split_text(retrieved_text)

# Step 4: LLM Processing
llm_start = time.time()
response = rag_chain.invoke({"query": query, "context": split_chunks[0]})
llm_end = time.time()
print(f"LLM processing time: {llm_end - llm_start:.2f} seconds")

# Step 5: Total time taken
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

# Print Answer
print("\n🔹 Answer:\n", response["result"])
print("\n📖 Sources:")
for doc in response["source_documents"]:
    print(f"- {doc.metadata['source']} (Page {doc.metadata['page']})\n")
