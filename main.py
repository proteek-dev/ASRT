import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import pickle

# Load models and initialize embeddings
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    qa_pipeline = pipeline("question-answering",
                            model="distilbert-base-cased-distilled-squad")
    return embeddings, qa_pipeline

embeddings, qa_pipeline = load_models()

# Sidebar for input
st.sidebar.title("Scheme Research Tool")
st.sidebar.markdown("**Instructions:**\n- Enter URLs or upload a text \
                    file containing URLs.\n- Click `Process URLs` \
                    to analyze.\n- Interact with the bot in the chat area.")

# URL Input Section
urls = st.sidebar.text_area("Enter URLs (one per line):")

# File Upload Section
uploaded_file = st.sidebar.file_uploader("Upload a text file containing URLs:",
                                          type=["txt"])

# Process URLs Button
process_button = st.sidebar.button("Process URLs")

# Main Section
st.title("Automated Scheme Research Tool")
st.markdown(
    """
    This chatbot allows you to interact with government scheme \
    articles based on key aspects:
    - Scheme Benefits
    - Application Process
    - Eligibility
    - Documents Required
    """
)

# Global variables for FAISS vector store
vectorstore = None
docs = []

# Session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Helper Function: Read URLs from Text File
def read_urls_from_file(uploaded_file):
    try:
        return uploaded_file.read().decode("utf-8").splitlines()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return []

# Helper Function: Process URLs
def process_urls(url_list):
    global vectorstore, docs
    st.info("Processing URLs...")
    loader = UnstructuredURLLoader(url_list)
    documents = loader.load()

    # Store documents and create embeddings
    docs.extend(documents)
    texts = [doc.page_content for doc in documents]
    vectorstore = FAISS.from_texts(texts, embeddings)

    # Save FAISS index and documents
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump((vectorstore, docs), f)
    st.success(f"Processed and saved {len(documents)} URLs!")

# Handle Processing of URLs
if process_button:
    url_list = []

    # Gather URLs from text area and uploaded file
    if urls.strip():
        url_list.extend(urls.strip().splitlines())
    if uploaded_file:
        file_urls = read_urls_from_file(uploaded_file)
        url_list.extend(file_urls)

    if url_list:
        process_urls(url_list)
    else:
        st.error("No URLs provided. Please enter URLs or \
                 upload a valid text file.")

# Load FAISS Index from File
def load_faiss_index():
    global vectorstore, docs
    try:
        with open("faiss_store.pkl", "rb") as f:
            vectorstore, docs = pickle.load(f)
        st.success("FAISS index loaded successfully!")
    except FileNotFoundError:
        st.info("No saved FAISS index found. Please process URLs first.")

# Automatically Load the Saved Index
load_faiss_index()

# Chatbot Interaction Section
st.subheader("Chat with the Scheme Research Bot")
query = st.text_input("You:", key="user_query")

if st.button("Send", key="send_query") and vectorstore and query:
    # Search the FAISS vector store
    result = vectorstore.similarity_search(query, k=1)
    if result:
        relevant_doc = result[0]
        context = relevant_doc.page_content
        answer = qa_pipeline(question=query, context=context)

        # Generate a short summary (first 200 characters) of the article
        summary = context[:200] + "..." if len(context) > 200 else context

        # Append response to chat history
        st.session_state.chat_history.append({
            "user": query,
            "bot": f"**Answer:** {answer['answer']}\n\n\
            **Source URL:** {relevant_doc.metadata.get('source', 'N/A')}\n\n\
            **Summary:** {summary}"
        })
    else:
        st.session_state.chat_history.append({
            "user": query,
            "bot": "I'm sorry, I couldn't find any relevant information."
        })

# Display Conversation History
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")

