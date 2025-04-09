from flask import Flask, render_template, request, redirect, url_for, session
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load LLM and FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
kb = FAISS.load_local("VD/FAISS", embedding_model, allow_dangerous_deserialization=True)
llm = OllamaLLM(model="mistral", max_tokens=100)
retriever = kb.as_retriever(search_kwargs={"k": 2})

# Prompt Template
def get_prompt(dosha):
    template = f"""
You are Ayurbot, an expert assistant in Ayurvedic remedies.
The user has a {dosha} dosha constitution.
Use the following context to answer the user's question with clarity and care. Do NOT reference chapters, appendices, or page numbers. Your answers should be self-contained, practical, and rooted in Ayurvedic principles. Avoid phrases like "explained in Chapter 6" or "described in Appendix 4".
If relevant, include suggestions on diet, lifestyle, yoga, breathwork, herbs, or daily routines. Keep the tone supportive and informative.

Context:
{{context}}

Question:
{{question}}

Helpful Answer:
"""
    return PromptTemplate(input_variables=["context", "question"], template=template)

def create_chain(dosha):
    prompt = get_prompt(dosha)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

@app.route('/')
def home():
    return render_template('portfolio.html')

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    questions = [
        "What is your body frame like?",
        "How is your appetite?",
        "How do you usually feel emotionally?",
        "How is your sleep pattern?",
        "How would you describe your energy levels?"
    ]
    options = [
        ["Light and thin", "Medium and muscular", "Heavy and solid"],
        ["Irregular, sometimes high, sometimes low", "Strong and consistent", "Slow but steady"],
        ["Anxious or restless", "Irritable or intense", "Calm and stable"],
        ["Light sleeper, wake easily", "Moderate sleeper", "Heavy sleeper"],
        ["Quick bursts of energy, then fatigue", "High energy, sometimes burnout", "Slow but sustained energy"]
    ]

    if request.method == "POST":
        answers = [request.form.get(f'q{i}') for i in range(1, 6)]
        print("Submitted answers:", answers)

        dosha_count = {'vata': 0, 'pitta': 0, 'kapha': 0}
        for ans in answers:
            if ans == 'a':
                dosha_count['vata'] += 1
            elif ans == 'b':
                dosha_count['pitta'] += 1
            elif ans == 'c':
                dosha_count['kapha'] += 1

        final_dosha = max(dosha_count, key=dosha_count.get)
        session['dosha'] = final_dosha  # Save dosha in session
        session['messages'] = []        # Reset chat history
        return redirect(url_for('chat'))

    return render_template("quiz.html", questions=questions, options=options)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    dosha = session.get('dosha')
    if not dosha:
        return redirect(url_for('quiz'))  # Redirect to quiz if not set

    if 'messages' not in session:
        session['messages'] = []

    messages = session['messages']

    if request.method == 'POST':
        question = request.form['message']
        chain = create_chain(dosha)
        response = chain.invoke({"query": question})
        answer = response['result']

        messages.append(("You", question))
        messages.append(("Ayurbot", answer))
        session['messages'] = messages

    return render_template('chat.html', dosha=dosha, messages=messages)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
