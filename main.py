from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import pickle
import faiss
import time 
import streamlit as st
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

st.title("Article Query Tool")
st.sidebar.title("Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

main_placeholder = st.empty()

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite', temperature=0.4)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...")
    docs = text_splitter.split_documents(data)

    main_placeholder.text("Embedding Vector...Started...")
    time.sleep(2)

    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    # docs_embeddings = embedding_model.embed_documents([doc.page_content for doc in docs])

    # embeddings_np = np.array(docs_embeddings, dtype=np.float32)

    # dimension = embeddings_np.shape[1]  
    # index = faiss.IndexFlatL2(dimension)
    # index.add(embeddings_np)

    vectorstore = FAISS.from_documents(docs, embedding_model)
    # retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
            result = chain({"question": query}, return_only_outputs=True)
            
            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
