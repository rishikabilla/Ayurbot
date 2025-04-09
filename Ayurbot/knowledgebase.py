import re
import fitz
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#STEP 1 : Load the PDF

#The PDF contains some pages which are images from which the text cannot be extracted using PyPDFLoader.
#Here,we are defining a function to identify such pages
def is_image_only(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    text = page.get_text()
    images = page.get_images()
    
    return not text.strip() and images 
#Defining the path of the pdf
Path="PDF/"
#loading the pdf
loader=DirectoryLoader(
    Path,
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
#creating a list of document objects where each element represents a single page from pdf
documents=loader.load()
valid_documents = []
#Here, we are iterating through each page in the documents(list) and checking if the page is not an image.
#If the page is not an image,then we append it to valid_documents(list).
for doc in documents:
    pdf_path = doc.metadata['source']  
    page_num = doc.metadata['page']  
    if not is_image_only(pdf_path, page_num):  
        valid_documents.append(doc)
#The below line prints the number of valid pages(except image pages).
#It is done to check whether everything is working well.
#print(f"Removed {len(documents) - len(valid_documents)} image-only pages.")
#Further,we want to convert this text into embeddings.Extra spaces will lead to wrong meanings.
#Here,we are defining a function to remove extra spaces from the extracted text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text
#Here,we are applying cleaning to all the pages
cleaned_documents = [Document(page_content=clean_text(doc.page_content), metadata=doc.metadata) for doc in valid_documents]
#The below lines prints the length of cleaned_documents(list) and valid_documents(list).
#It is done to check whether everything is working properly.
#print(len(valid_documents))
#print(len(cleaned_documents))

#STEP 2 : Create Chunks

#Here,we are splitting the extracted text into chunks, as processing through parts(chunks) is efficient than scanning through entire data.
#Chunk overlap is the number of characters at the end of prev chunk that will be repeated in the next chunk so that context is maintained.
text_splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=200)
text_chunks=text_splitter.split_documents(cleaned_documents)
#The below lines prints the length of text_chunks.
#It is done to check whether everything is working properly.
#print(len(text_chunks))


#STEP 3: Create Embeddings

#Here,we are using a model named "all-MiniLM-L6-v2" to convert our chunks into embeddings.
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#specifying the path to store the vector embeddings locally
KB_path="VD/FAISS"
#FAISS is an vector database in which we can store and retrieve embeddings.
kb=FAISS.from_documents(text_chunks,embedding_model)
#Saving the FAISS vector database locally
kb.save_local(KB_path)