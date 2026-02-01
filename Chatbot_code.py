# ========== CELL 3: SAVE WORKING ROCKYBOT ==========
%%writefile rockybot_fixed.py

import os
import streamlit as st
import pickle
import time
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Validate URLs
    valid_urls = [url for url in urls if url.strip()]
    if not valid_urls:
        st.error("Please enter at least one valid URL")
        st.stop()
    
    # Load data
    loader = UnstructuredURLLoader(urls=valid_urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    # Create embeddings and save to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)
    
    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)
    
    main_placeholder.text("Processing Complete! ✅ You can now ask questions.")

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            
            # ✅ CUSTOM RETRIEVAL QA IMPLEMENTATION
            # Get relevant documents
            docs = vectorstore.similarity_search(query, k=4)
            
            # Extract context and sources
            context = "\n\n".join([doc.page_content for doc in docs])
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))
            
            # Create prompt template
            prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Helpful Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            
            # Generate answer
            final_prompt = prompt.format(context=context, question=query)
            
            with st.spinner("Generating answer..."):
                answer = llm.invoke(final_prompt)
            
            # Display results
            st.header("Answer")
            st.write(answer)
            
            # Display sources
            if sources and sources != ['Unknown']:
                st.subheader("Sources:")
                for source in sources:
                    if source != 'Unknown':
                        st.write(f"• {source}")
    else:
        st.warning("⚠️ Please process URLs first by clicking 'Process URLs' button.")

print("✅ File 'rockybot_fixed.py' created successfully!")