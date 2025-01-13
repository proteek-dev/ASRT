### README.md: 
**Automated Scheme Research Tool**

---

#### **Table of Contents**
1. [Introduction](#introduction)
2. [Features](#features)
3. [Code Structure](#code-structure)
4. [Installation](#installation)
5. [Usage Instructions](#usage-instructions)
6. [Technical Details](#technical-details)
7. [Solution Acceptance Criteria](#solution-acceptance-criteria)
8. [Future Enhancements](#future-enhancements)

---

### **Introduction**

The **Automated Scheme Research Tool** is a Streamlit-based web application that processes government scheme articles and enables users to:
- Summarize the articles based on key aspects such as **Benefits**, **Application Process**, **Eligibility**, and **Documents Required**.
- Interact with a chatbot to ask questions about the schemes.
- Receive answers, along with the source URL and a brief summary of the relevant article.

This tool leverages free, open-source libraries like **LangChain**, **HuggingFace Transformers**, **FAISS**, and **SentenceTransformers** for embeddings, indexing, and question-answering.

---

### **Features**

1. **Input Options**:
   - Enter URLs directly into a text area.
   - Upload a `.txt` file containing multiple URLs.

2. **Processing**:
   - Fetch content from URLs using `LangChain.UnstructuredURLLoader`.
   - Generate embeddings using `HuggingFace SentenceTransformers`.
   - Create a FAISS index for efficient similarity-based retrieval.

3. **Chatbot Interaction**:
   - Users can ask questions and receive:
     - Relevant answers.
     - Source URL of the article.
     - A concise summary of the article.

4. **Persistent Index**:
   - Automatically saves the FAISS index and loaded documents in a `faiss_store.pkl` file for future use.

5. **User-Friendly Interface**:
   - A responsive and interactive web interface using **Streamlit**.

---

### **Code Structure**

The project contains the following components:

| File/Folder          | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `main.py`            | The main Python script containing the Streamlit application logic.         |
| `requirements.txt`   | List of all the Python libraries required for the project.                 |
| `faiss_store.pkl`    | A pickle file storing the FAISS index and document metadata (created at runtime). |
| `.config` (Optional) | Configuration file (if needed for custom settings).                        |

---

### **Installation**

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run main.py
   ```

---

### **Usage Instructions**

1. **Provide URLs**:
   - Enter one or more URLs in the sidebar text area.
   - Or upload a `.txt` file containing URLs (one URL per line).

2. **Process URLs**:
   - Click the "Process URLs" button to fetch and analyze the content.
   - Wait for the success message confirming the processing.

3. **Interact with the Chatbot**:
   - Type a question in the input box and click "Send."
   - View the bot’s response, including the answer, source URL, and article summary.

4. **Revisit Processed Data**:
   - If URLs have been previously processed, the FAISS index will be automatically loaded.

---

### **Technical Details**

1. **Libraries Used**:
   - **Streamlit**: For building the web application.
   - **LangChain**: For document loading and pre-processing.
   - **SentenceTransformers**: For generating embedding vectors.
   - **FAISS**: For fast similarity-based search.
   - **HuggingFace Transformers**: For question-answering tasks.
   - **BeautifulSoup**: For HTML parsing and content extraction.

2. **Workflow**:
   - **Data Loading**: URLs are fetched and parsed using `UnstructuredURLLoader`.
   - **Embeddings**: Texts are embedded using SentenceTransformers (`all-MiniLM-L6-v2`).
   - **Indexing**: FAISS indexes the embeddings for efficient similarity search.
   - **QA Interaction**: HuggingFace’s `distilbert-base-cased-distilled-squad` model provides answers to user queries.

3. **File Persistence**:
   - The FAISS index and document metadata are saved as `faiss_store.pkl` for reusability.

---

### **Solution Acceptance Criteria**

| **Criteria**                                | **Met** |
|--------------------------------------------|---------|
| Web app opens in a browser                 | ✅       |
| Users can input URLs directly              | ✅       |
| Users can upload text files containing URLs| ✅       |
| Data is processed with LangChain           | ✅       |
| Embedding vectors are generated (free)     | ✅       |
| FAISS indexes are used for retrieval       | ✅       |
| Users can ask questions and get answers    | ✅       |
| Answers include source URL and summaries   | ✅       |
| FAISS index is saved in a local pickle file| ✅       |

---

### **Future Enhancements**

1. **Multi-Language Support**:
   - Incorporate multilingual models for handling non-English scheme articles.

2. **Batch Processing**:
   - Enable batch processing for large-scale data ingestion.

3. **Advanced Summarization**:
   - Add advanced summarization techniques for more detailed article summaries.

4. **Enhanced UI**:
   - Improve the chatbot interface for a richer user experience.

---

### **Comments in Code**

Each section of the code is well-commented to explain:
- The purpose of the function or block.
- Input and output handling.
- Integration of libraries and workflows.

For example:

```python
# Helper Function: Process URLs
def process_urls(url_list):
    global vectorstore, docs
    st.info("Processing URLs...")
    loader = UnstructuredURLLoader(url_list)  # Load content from URLs
    documents = loader.load()

    # Store documents and create embeddings
    docs.extend(documents)
    texts = [doc.page_content for doc in documents]
    vectorstore = FAISS.from_texts(texts, embeddings)  # Create FAISS index

    # Save FAISS index and documents
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump((vectorstore, docs), f)
    st.success(f"Processed and saved {len(documents)} URLs!")
```

---

This README should guide users and developers through setting up, running, and understanding the Automated Scheme Research Tool. Let me know if additional details are needed!