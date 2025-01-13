**Automated Scheme Research Tool**

---

#### **Table of Contents**
1. [Introduction](#introduction)
2. [Features](#features)
3. [Code Structure](#code-structure)
4. [Installation](#installation)
5. [Usage Instructions](#usage-instructions)
6. [How FAISS Works](#how-faiss-works)
7. [How the Summary and Chat Feature Works](#how-the-summary-and-chat-feature-works)
8. [Code Explanation](#code-explanation)
9. [Solution Acceptance Criteria](#solution-acceptance-criteria)
10. [Future Enhancements](#future-enhancements)

---

### **Introduction**

The **Automated Scheme Research Tool** is a chatbot-driven web application that processes government scheme articles and provides interactive query capabilities. It uses **FAISS** for fast retrieval, **HuggingFace Transformers** for question-answering, and **LangChain** for document handling.

---

### **Features**

- Input URLs manually or via file upload.
- Summarize schemes into:
  - Benefits
  - Application Process
  - Eligibility
  - Documents Required
- Ask questions about schemes and receive:
  - A precise answer.
  - Source URL of the scheme.
  - A brief summary of the article.
- Save and load a persistent FAISS index for reusability.

---

### **Code Structure**

| File/Folder          | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `main.py`            | Contains the Streamlit app logic.                                          |
| `requirements.txt`   | List of Python libraries required for the project.                        |
| `faiss_store.pkl`    | Stores the FAISS index and document metadata.                             |
| `.config` (Optional) | Configuration file for custom settings.                                   |

---

### **Installation**

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Set up a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run main.py
   ```

---

### **Usage Instructions**

1. Enter URLs or upload a `.txt` file with URLs.
2. Click "Process URLs" to fetch and analyze the content.
3. Ask a question in the chat area.
4. Receive:
   - Answer based on the article content.
   - The article’s URL.
   - A short summary of the relevant article.

---

### **How FAISS Works**

#### **Overview**
FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It is ideal for retrieving information in large datasets by leveraging vector embeddings.

#### **Steps in the Application**:
1. **Create Embeddings**:
   - Each scheme article's content is embedded into a dense vector using **HuggingFace SentenceTransformers**.
   - These embeddings are numerical representations of the text.

2. **Indexing**:
   - The dense vectors are stored in a FAISS index for fast retrieval.
   - FAISS uses an approximate nearest neighbor (ANN) search algorithm for high-speed similarity searches.

3. **Query Handling**:
   - User queries are also embedded into vectors.
   - FAISS compares the query vector with the indexed vectors to find the most similar content.

4. **Output**:
   - The most relevant document is retrieved and used to generate a response.

#### **Advantages of FAISS**:
- Speed: Handles large datasets efficiently.
- Scalability: Can scale to millions of documents.
- Flexibility: Supports multiple similarity measures (e.g., cosine similarity).

---

### **How the Summary and Chat Feature Works**

#### **Summary Generation**:
1. The first 200 characters of the article content are extracted as a **brief summary**.
2. If the content is shorter than 200 characters, the full text is used.

#### **Chat Interaction**:
1. **User Query**:
   - The user inputs a question into the chat interface.
2. **Retrieval**:
   - FAISS retrieves the most relevant document based on similarity with the query embedding.
3. **Question-Answering**:
   - The content of the retrieved document is passed to a **HuggingFace QA model**.
   - The QA model extracts the most likely answer from the document.
4. **Response**:
   - The bot responds with:
     - **Answer**: The extracted answer.
     - **URL**: The source of the document.
     - **Summary**: A short overview of the document.

---

### **Code Explanation**

#### **FAISS Integration**:
```python
# Create embeddings and FAISS index
texts = [doc.page_content for doc in documents]
vectorstore = FAISS.from_texts(texts, embeddings)  # Create FAISS index
with open("faiss_store.pkl", "wb") as f:
    pickle.dump((vectorstore, docs), f)  # Save FAISS index and documents
```
- **`FAISS.from_texts`**: Converts documents into embeddings and creates the FAISS index.
- **Pickle**: Saves the index and documents for reusability.

#### **Processing User Queries**:
```python
result = vectorstore.similarity_search(query, k=1)  # Retrieve most similar document
if result:
    relevant_doc = result[0]
    context = relevant_doc.page_content
    answer = qa_pipeline(question=query, context=context)  # Generate answer
    summary = context[:200] + "..." if len(context) > 200 else context  # Short summary
```
- **`similarity_search`**: Finds the document most similar to the query.
- **HuggingFace QA Pipeline**: Extracts the best answer from the document.

#### **Displaying the Response**:
```python
st.session_state.chat_history.append({
    "user": query,
    "bot": f"**Answer:** {answer['answer']}\n\n**Source URL:** {relevant_doc.metadata.get('source', 'N/A')}\n\n**Summary:** {summary}"
})
```
- Appends the user's query and bot's response (answer, URL, summary) to the chat history.

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

- **Topic Categorization**:
  - Automatically categorize scheme articles by type (e.g., healthcare, education).
- **Improved Summarization**:
  - Use advanced summarization models for detailed overviews.
- **Real-Time Indexing**:
  - Allow users to add URLs and update the FAISS index dynamically.

---