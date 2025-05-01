# Q&A Chatbot
from langchain import HuggingFaceHub
from dotenv import load_dotenv
import streamlit as st
import os
from huggingface_hub import HfApi

# Load environment variables from .env
load_dotenv()

# Function to load Hugging Face model and get responses
def get_huggingface_response(question):
    # Initialize the Hugging Face model
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",  # Specify the model repository
        task="text2text-generation",  # Explicitly specify the task
        model_kwargs={"temperature": 0.5, "max_length": 64}
    )
    response = llm(question)
    return response

# Initialize our Streamlit app
st.set_page_config(page_title="Q&A Demo")
st.header("Langchain Application")

# Input and response
input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

# If ask button is clicked
if submit:
    response = get_huggingface_response(input)
    st.subheader("The Response is")
    st.write(response)
    
api = HfApi()
api_token = os.getenv("HUGGINGFACE_API_KEY")
if api_token:
    print("API key is set correctly.")
else:
    print("API key is missing or invalid.")