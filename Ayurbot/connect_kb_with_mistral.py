from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Load vector database
def load_VD():
    KB_path = "VD/FAISS"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    kb = FAISS.load_local(KB_path, embedding_model, allow_dangerous_deserialization=True)
    return kb

# Load local LLM
def load_llm():
    llm = OllamaLLM(model="mistral", max_tokens=100)
    return llm

# Ask user to determine their dosha type
def determine_dosha():
    print("Before we begin, let's determine your dosha type.\nPlease answer the following questions:\n")
    vata, pitta, kapha = 0, 0, 0

    q1 = input("1. Do you usually feel (a) cold, (b) warm, or (c) moderate? ").lower()
    if 'a' in q1: vata += 1
    elif 'b' in q1: pitta += 1
    elif 'c' in q1: kapha += 1

    q2 = input("2. Your energy is (a) irregular and quick to tire, (b) strong and intense, (c) steady and slow? ").lower()
    if 'a' in q2: vata += 1
    elif 'b' in q2: pitta += 1
    elif 'c' in q2: kapha += 1

    q3 = input("3. Your digestion is (a) irregular/sensitive, (b) strong/fast, (c) slow/heavy? ").lower()
    if 'a' in q3: vata += 1
    elif 'b' in q3: pitta += 1
    elif 'c' in q3: kapha += 1

    q4 = input("4. Your skin is (a) dry/rough, (b) sensitive/rashy, (c) oily/thick? ").lower()
    if 'a' in q4: vata += 1
    elif 'b' in q4: pitta += 1
    elif 'c' in q4: kapha += 1

    q5 = input("5. You sleep (a) lightly/wake easily, (b) moderately, (c) deeply/for long hours? ").lower()
    if 'a' in q5: vata += 1
    elif 'b' in q5: pitta += 1
    elif 'c' in q5: kapha += 1

    scores = {"Vata": vata, "Pitta": pitta, "Kapha": kapha}
    dominant_dosha = max(scores, key=scores.get)
    print(f"\n🩺 Based on your answers, your dominant dosha appears to be: **{dominant_dosha}**\n")
    return dominant_dosha

def main():
    kb = load_VD()
    llm = load_llm()
    retriever = kb.as_retriever(search_kwargs={"k": 2})
    dosha = determine_dosha()

    prompt_template = """
You are Ayurbot, an expert assistant in Ayurvedic remedies.
The user has a {dosha_type} dosha constitution.
Use the following context to answer the user's question with clarity and care. Do NOT reference chapters, appendices, or page numbers. Your answers should be self-contained, practical, and rooted in Ayurvedic principles. Avoid phrases like "explained in Chapter 6" or "described in Appendix 4".
If relevant, include suggestions on diet, lifestyle, yoga, breathwork, herbs, or daily routines. Keep the tone supportive and informative.

Context:
{context}

Question:
{question}

Helpful Answer:
"""

    # Build custom prompt by injecting dosha type into prompt context
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template.replace("{dosha_type}", dosha),
    )

    def dosha_rag_chain(query):
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": custom_prompt}
        ).invoke({"query": query})

    print("🌿 Welcome to Ayurbot! Type your question or type 'exit' to quit.\n")

    while True:
        query = input("Ask Ayurbot: ").strip()
        if query.lower() in ['exit', 'quit']:
            print("👋 Goodbye! Stay healthy with Ayurveda!")
            break
        response = dosha_rag_chain(query)
        print("\n💡 Answer:\n", response["result"], "\n")

if __name__ == "__main__":
    main()
