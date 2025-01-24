import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import SentenceTransformerEmbeddings  # Import Sentence-Transformers embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import time

from dotenv import load_dotenv
load_dotenv()

# Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']
urls = [
    "https://www.datanetiix.com/leadership.php",
    "https://www.datanetiix.com/artificial-intelligence-and-machine-learning.php",
    "https://www.datanetiix.com/mobile-apps.php", 
    "https://www.datanetiix.com/wearable-app-development.php"
]

# Initialize Sentence-Transformers embeddings
if "vector" not in st.session_state:
    st.session_state.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")  # Using pre-trained model

    # Initialize document loader and load documents from multiple URLs
    st.session_state.docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        st.session_state.docs.extend(loader.load())  # Add loaded documents to the session state

    # Split the loaded documents into 
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    # Create FAISS vector store
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("WebSite Link Based -Chatbot Datanetiix")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions:{input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt_input = st.text_input("Input your prompt here")

if prompt_input:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt_input})
    print("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            # st.write(doc.page_content)
            st.write("--------------------------------")
