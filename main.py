import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY") if "GROQ_API_KEY" in os.environ else st.secrets["GROQ_API_KEY"]
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY") if "LANGCHAIN_API_KEY" in os.environ else st.secrets["LANGCHAIN_API_KEY"]


os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Create the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked"),
        ("user", "Question:{question}")
    ]
)

# Streamlit UI
st.set_page_config(page_title="Conversational AI chatbot", page_icon="🦜")
st.title(body="Conversational AI chatbot")
input_text = st.text_input(label="What question you have in mind?")

# Set up the LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")
output_parser = StrOutputParser()
chain = prompt_template | llm | output_parser

# Process user input
if input_text:
    st.write(chain.invoke({"question": input_text}))
