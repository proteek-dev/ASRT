import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import pickle

# Load models
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return embeddings, qa_pipeline

embeddings, qa_pipeline = load_models()

# Sidebar: Input URLs
st.sidebar.title("Scheme Research Tool")
urls = st.sidebar.text_area("Enter URLs (one per line):")
process_button = st.sidebar.button("Process URLs")

# Main section
st.title("Automated Scheme Research Tool")
st.write("Summarize and ask questions about government schemes!")

# Initialize variables
vectorstore = None
docs = []

# Helper: Process URLs
def process_urls(urls):
    global vectorstore, docs
    st.info("Processing URLs...")
    url_list = urls.splitlines()
    loader = UnstructuredURLLoader(url_list)
    documents = loader.load()
    
    docs.extend(documents)
    texts = [doc.page_content for doc in documents]
    
    # Create embeddings and FAISS index
    vectorstore = FAISS.from_texts(texts, embeddings)
    st.success(f"Processed {len(documents)} documents!")

# Process URLs when button clicked
if process_button and urls:
    process_urls(urls)

# Query section
query = st.text_input("Ask a question:")
if st.button("Submit Query") and vectorstore and query:
    # Search for relevant document
    result = vectorstore.similarity_search(query, k=1)
    
    if result:
        context = result[0].page_content
        answer = qa_pipeline(question=query, context=context)
        
        st.write(f"**Answer:** {answer['answer']}")
        st.write(f"**Relevant Context:** {context[:500]}...")
    else:
        st.error("No matching documents found.")

# Save FAISS index
if st.sidebar.button("Save Index"):
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump((vectorstore, docs), f)
    st.success("Index saved!")
