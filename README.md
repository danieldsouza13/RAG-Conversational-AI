# RAG Application with Cohere LLM and MongoDB Atlas Retriever

This project is a Retrieval-Augmented Generation (RAG) application using LangChain Community, MongoDB Atlas, and the Cohere LLM model. It demonstrates how to ingest documents, create embeddings, and use them to answer queries with a language model.

## Setup Instructions

### Prerequisites

    - Python 3.8 or higher
    - MongoDB Atlas account
    - Cohere API key

### 1. Clone the Repository

    - ```sh
    - git clone <your_github_repo_url>
    - cd basic-rag-mongodb-cohere

### 2. Install Dependencies

    - pip install - r requirements.txt

### 3. Modify 'params.py'

    - Update with your Cohere API key, MongoDB connection string, database name, collection name, and index name.

### 4. Run the  ingest documents script to ingest sample documents into your MongoDB Atlas collection

    - python ingest_docs.py

### 5. Run the RAG Application

    - python rag_app.py

### 6. Enter your query

    - Enter your query in the terminal when prompted. The application will retrieve relevant documents and generate an answer using the Cohere LLM model.

 ### 7. Evaluate the Application

    - python eval_script.py
    - the evaluatuon script will print precision, recall, and F1-score based on the provided test queries and ground truths.



