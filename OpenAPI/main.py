import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import openai
import configparser
import pickle

# Load API key from .config file
def load_openai_key():
    config = configparser.ConfigParser()
    config.read(".config")
    return config["openai"]["OPENAI_API_KEY"]

# Set OpenAI API key
openai.api_key = load_openai_key()

# Load OpenAI Embeddings
@st.cache_resource
def load_embeddings():
    return OpenAIEmbeddings()

embeddings = load_embeddings()

# Sidebar for input
st.sidebar.title("Automated Scheme Research Tool")
st.sidebar.markdown("**Instructions:**\n- Enter URLs or upload \
                    a text file containing URLs.\n- Click `Process URLs` \
                    to analyze.\n- Interact with the bot in the chat area.")

# URL Input Section
urls = st.sidebar.text_area("Enter URLs (one per line):")

# File Upload Section
uploaded_file = st.sidebar.file_uploader("Upload a text file containing URLs:",
                                          type=["txt"])

# Process URLs Button
process_button = st.sidebar.button("Process URLs")

# Main Section
st.title("Scheme Research Chatbot")
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
    with open("faiss_store_openai.pkl", "wb") as f:
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
        with open("faiss_store_openai.pkl", "rb") as f:
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
        matched_doc = result[0]
        context = matched_doc.page_content
        answer = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Provide a detailed answer to the question:
              '{query}' based on the following context:\n\n{context}",
            max_tokens=200
        )["choices"][0]["text"].strip()

        summary = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Summarize the following text:\n\n{context}",
            max_tokens=100
        )["choices"][0]["text"].strip()

        # Append to chat history
        st.session_state.chat_history.append({
            "user": query,
            "bot": {
                "answer": answer,
                "url": matched_doc.metadata["source"],
                "summary": summary
            }
        })
    else:
        st.session_state.chat_history.append({
            "user": query,
            "bot": "I'm sorry, I couldn't find any relevant information."
        })

# Display Conversation History
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    if isinstance(chat["bot"], dict):
        st.markdown(f"**Bot:** {chat['bot']['answer']}")
        st.markdown(f"**Source URL:** {chat['bot']['url']}")
        st.markdown(f"**Summary:** {chat['bot']['summary']}")
    else:
        st.markdown(f"**Bot:** {chat['bot']}")
