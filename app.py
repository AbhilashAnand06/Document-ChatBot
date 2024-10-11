import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS # vector store DB
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings # vector embedding technique
from dotenv import load_dotenv

import time

load_dotenv()

## load the GROQ And GOOGLE API KEY from the .env file 
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.title("Document ChatBot")
st.subheader("Powered by Gemma model")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="gemma-7b-it")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./data") # Data Ingestion
        st.session_state.docs=st.session_state.loader.load() # Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) # Chunk Creation
        st.session_state.final_document=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) # Splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_document,st.session_state.embeddings) # Vector OpenAI embeddings




st.write("Click the button to embed documents and load the vector store DB")
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector store DB is ready")

prompt1=st.text_input("Enter your question: ")

if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    st.write(response['answer'])
    response_time = time.process_time()-start
    st.write(f"Response time : {response_time:.2f} seconds")

    # With a streamlit expander
    with st.expander("Document Similarity Search (view information sources)"):
        # Find the relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("--------------------------------")
