from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
 #loading the vector database
KB_path="KnowledgeBase/FAISS_KB"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
kb=FAISS.load_local(KB_path,embedding_model,allow_dangerous_deserialization=True)
retriever = kb.as_retriever(search_kwargs={"k": 2})


# Placeholder for LLM (will be added after download)
llm = OllamaLLM(model="mistral", max_tokens=256)  



# Create a RAG pipeline (waiting for LLM)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,  # This will be updated later
    chain_type="stuff",  # Directly insert retrieved text into prompt
    retriever=retriever,
    return_source_documents=True
)

query = input("Ask Ayurbot: ")
response = rag_chain.invoke({"query": query})


print("\n🔹 Answer:\n", response["result"])
print("\n📖 Sources:")
for doc in response["source_documents"]:
    print(f"- {doc.metadata['source']} (Page {doc.metadata['page']})\n")
