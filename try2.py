from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Define FAISS database path
KB_path = "KnowledgeBase/FAISS_KB"

# Load embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Correctly load FAISS index in LangChain format
kb = FAISS.load_local(KB_path, embedding_model, allow_dangerous_deserialization=True)

# Convert FAISS to retriever
retriever = kb.as_retriever(search_kwargs={"k": 2})

# Load the LLM model
llm = OllamaLLM(model="mistral", max_tokens=256)

# Create RAG pipeline
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Get user query
query = input("Ask Ayurbot: ")
response = rag_chain.invoke({"query": query})

# Print response
print("\n🔹 Answer:\n", response["result"])
print("\n📖 Sources:")
for doc in response["source_documents"]:
    print(f"- {doc.metadata['source']} (Page {doc.metadata['page']})\n")
