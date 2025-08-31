from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()
persist_dir = "db/chroma-pdf"

st.title("PDF Q&A App")
st.header("Welcome to the AI PDF Q&A App")

pdf_is_stored = False

def store_pdf():
    global pdf_is_stored
    select_pdf_btn = st.file_uploader("Upload PDF file", type=["pdf"])

    try:
        if select_pdf_btn:
            binary_data = select_pdf_btn.getvalue()
            pdf_viewer(binary_data, width=1000, height=500)

            if st.button("Save to Vector DB"):
                # save uploaded PDF temporarily
                temp_path = "temp.pdf"
                with open(temp_path, "wb") as f:
                    f.write(binary_data)

                # load PDF
                pdf = PyPDFLoader(temp_path)
                read_pdf = pdf.load()

                # split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                docs = text_splitter.split_documents(read_pdf)

                # embeddings model
                # embeddings = OllamaEmbeddings(model="llama3.1")
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

                # store in Chroma
                Chroma.from_documents(
                    docs, embeddings, persist_directory=persist_dir
                )
            

                st.success("✅ Vector DB saved successfully!")
                st.session_state.pdf_is_stored = True

    except Exception as e:
        st.error(f"Error: {e}")

def talk_to_llm():
    if st.session_state.pdf_is_stored:
        # query input
        query = st.text_input("Enter your question: ")
        
        # embeddings model
        # embeddings = OllamaEmbeddings(model="llama3.1")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # load vector DB
        vector_store = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
        
        
        
        result = vector_store.similarity_search(query, k=3)
        
        
        if query:
            btn = st.button("Ask")
            if btn:
                # chatbot
                llm = ChatOllama(
                    model="llama3.1",
                    temperature=0,
                    # other params...
                )
                
                chat = ChatPromptTemplate([
                    ("system", """You are a helpful AI assistant who answers questions based on a provided document. If the answer is not in the document, respond with 'I don't know'.
                     
                            Here is the document:
                            
                            {pdf}
                     """),
                    ("human", "{query}")
                ])
                
                chain = chat | llm | StrOutputParser()
                
                response = chain.invoke({"query": query, "pdf": [result.page_content for result in result]})


                
                st.write(response)
                # st.write([result.page_content for result in result])
    

if __name__ == "__main__":
    store_pdf()
    talk_to_llm()