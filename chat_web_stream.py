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

# Load the GROQ API key from Streamlit secrets
groq_api_key = st.secrets["api_keys"]["GROQ_API_KEY"]

# Your code for loading documents, embeddings, and processing remains the same
urls = [
"https://www.britannica.com/list/17-questions-about-health-and-wellness-answered"
]

# Initialize Sentence-Transformers embeddings
if "vector" not in st.session_state:
    st.session_state.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")  # Using pre-trained model

    # Initialize document loader and load documents from multiple URLs
    st.session_state.docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        st.session_state.docs.extend(loader.load())  # Add loaded documents to the session state

    # Split the loaded documents into chunks
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    # Create FAISS vector store
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Title for the Streamlit app
st.title("AI Chatbot - Health_benifit")

# Initialize the Groq model with the API key
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Define the chat prompt
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Questions: {input}
""")

# Create the document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Text input for user prompt
prompt_input = st.text_input("Input your prompt here")

if prompt_input:
    # Track processing time
    start = time.process_time()
    
    # Get the response from the retrieval chain
    try:
        response = retrieval_chain.invoke({"input": prompt_input})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])
        
        # # Display relevant documents using an expander
        # with st.expander("Document Similarity Search"):
        #     if "context" in response:
        #         for i, doc in enumerate(response["context"]):
        #             st.write(f"Document {i+1}:")
        #             st.write(f"Snippet: {doc['page_content'][:300]}...")  # Display the first 300 chars of the document content
        #             st.write(f"URL: {urls[i]}")
        #             st.write("-" * 50)
        #     else:
        #         st.write("No context found for the provided input.")

    except Exception as e:
        st.error(f"Error during processing: {e}")
