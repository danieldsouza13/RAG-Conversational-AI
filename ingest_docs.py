import json
from cohere import Client as CohereClient
from pymongo import MongoClient
from params import COHERE_API_KEY, MONGODB_CONN_STRING, DB_NAME, DOCS_COLLECTION
import logging
import os

def ingest_documents():
    try:
        # Path for document file
        jsonl_file_path = os.path.join(os.path.dirname(__file__), 'Documents', 'manual_documents 1.jsonl')

        # Load documents from the manual JSONL file
        with open(jsonl_file_path, 'r') as file:
            documents = [json.loads(line) for line in file]

        # Extract content from documents
        contents = [doc['page_content'] for doc in documents]

        # Initialize Cohere client for embeddings
        cohere_client = CohereClient(COHERE_API_KEY)

        # Generate embeddings for document contents
        embeddings = cohere_client.embed(texts=contents).embeddings

        # Initialize MongoDB client
        mongo_client = MongoClient(MONGODB_CONN_STRING)
        db = mongo_client[DB_NAME]
        collection = db[DOCS_COLLECTION]

        # Store documents and embeddings in MongoDB
        for content, embedding in zip(contents, embeddings):
            collection.insert_one({
                'content': content,
                'embedding': embedding
            })

        logging.info("Documents ingested and stored successfully.")
        return contents

    except Exception as e:
        logging.error(f"Error ingesting documents: {e}")
        raise

if __name__ == "__main__":
    ingest_documents()
