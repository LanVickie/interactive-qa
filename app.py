import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import pysqlite3
import sys
import chromadb
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def generate_response(uploaded_file, openai_api_key, query_text,url_api_base):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, openai_api_base=url_api_base)
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # chromadb.api.client.SharedSystemClient.clear_system_cache()
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key, base_url=url_api_base), chain_type='stuff', retriever=retriever)
        return qa.run(query_text)

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')
# Query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    url_api_base = st.text_input('OpenAI Base URL', type='default', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text,url_api_base)
            chromadb.api.client.SharedSystemClient.clear_system_cache()
            result.append(response)
            del openai_api_key
            del url_api_base

if len(result):
    st.info(response)
