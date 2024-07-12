# RAG Conversational AI with Cohere LLM and MongoDB Atlas Retriever

This project is a Retrieval-Augmented Generation (RAG) Conversational AI using LangChain, MongoDB Atlas, and the Cohere LLM model. It demonstrates how to ingest documents, create embeddings, and use them to answer queries with a large language model.

## Setup Instructions

### Prerequisites (ALL FREE!)

    - Python 3.8 or higher
    - MongoDB Atlas account
    - Cohere API key
    - Langchain API Key

### 1. Clone the Repository

    - ```sh
    - git clone <your_github_repo_url>
    - cd RAG-Conversational-AI

### 2. Install Dependencies

    - pip install - r requirements.txt

### 3. Create a ".env" file in the root of your project directory

    - Follow the ".env.example" file to create a ".env" file in the root of your project directory. Replace the environment variable values with your own. "Params.py" will get the respective information from your ".env" file to ensure that your environment variables are protected.

### 4. Run the  ingest documents script to ingest sample documents into your MongoDB Atlas collection

    - python ingest_docs.py

### 5. Run the RAG Conversational AI Application

    - python main.py

### 6. Enter your query

    - Enter your query in the terminal when prompted. The application will first reshape your question if necessary and then proceed on to the Inner Route Decision. In this step the application will either utilize the RAG Application route to format a response based solely on relevant information from documents that you have ingested into your MongoDB database, or if no relevant information was found in your database it will follow the Chat Model route to produce its own response utilzing the Cohere LLM Model.

 ### 7. Evaluate the Application

    - python eval_script.py
    - the evaluation script will print precision, recall, and F1-score based on the provided test queries and ground truths.



