Ayurbot is an chatbot that delivers personalized Ayurvedic home remedies based on your body constitution (dosha). It uses a Retrieval-Augmented Generation (RAG) pipeline with traditional Ayurvedic texts to give holistic, dosha-specific solutions.

**---Features---**

**Dosha Diagnosis**

Take a quick 5-question quiz to determine your Ayurvedic body type: Vata, Pitta, or Kapha.
**Knowledge Base**

Retrieves accurate information from three renowned Ayurvedic books:

-->The Complete Book of Ayurvedic Home Remedies – Dr. Vasant Lad

-->Charaka Samhita - Acharya Charaka,Acharya Dridhabala

-->The Everyday Ayurveda Cookbook –  Kate O'Donnell, Cara Brostrom

**Personalized Remedies**

Provides natural home remedies tailored to your unique dosha for better health and balance.

**Interactive Chat UI**

A clean,chat-style interface with a black and green theme for a modern herbal vibe.

**Fast & Accurate Retrieval**

Uses FAISS vector database for fast semantic search across all books.

**Tech Stack**

| Layer        | Tools Used                             |
|--------------|----------------------------------------|
| Frontend     | HTML, CSS (custom black-green theme)   |
| Backend      | Flask (Python)                         |
| LLM          | Mistral via Ollama                     |
| Embeddings   | all-MiniLM-L6-v2(Hugging Face)         |
| Vector DB    | FAISS                                  |
| PDF Parsing  | PyMuPDF                                |
| Chunking     | LangChain's RecursiveCharacterTextSplitter |
| Prompting    | LangChain + RetrievalQA                |

**Screenshots**

#Portfolio Page

![image](https://github.com/user-attachments/assets/22d1e546-b171-4683-9b2d-e1a3a60121e6)

#Portfolio Page with button for quiz

![image](https://github.com/user-attachments/assets/b2ccae76-63f3-4885-8db1-53d37b47caf8)

#quiz page

![image](https://github.com/user-attachments/assets/5963cf59-612e-4812-9859-dd7095da8de9)
#chat page
![image](https://github.com/user-attachments/assets/4dcce4c6-25cc-45be-bc07-9f91aeaff623)




