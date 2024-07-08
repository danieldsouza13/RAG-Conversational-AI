from langchain_community.document_loaders import WebBaseLoader
from cohere import Client as CohereClient
from pymongo import MongoClient, ASCENDING
from params import cohere_api_key, mongodb_conn_string, db_name, collection_name
import logging

def ingest_documents():
    try:
        # Load documents from the web
        urls = [ 'https://en.wikipedia.org/wiki/AT&T','https://en.wikipedia.org/wiki/Bank_of_America']
        loader = WebBaseLoader(urls)
        documents = loader.load()

        # Extract content from documents
        contents = [doc.page_content for doc in documents]

        # Initialize Cohere client for embeddings
        cohere_client = CohereClient(cohere_api_key)

        # Generate embeddings for document contents
        embeddings = cohere_client.embed(texts=contents).embeddings

        # Initialize MongoDB client
        mongo_client = MongoClient(mongodb_conn_string)
        db = mongo_client[db_name]
        collection = db[collection_name]

        # Ensure unique index on content
        collection.create_index([('content', ASCENDING)], unique=True)

        # Store documents and embeddings in MongoDB if they don't already exist
        for content, embedding in zip(contents, embeddings):
            try:
                collection.update_one(
                    {'content': content},
                    {'$set': {'content': content, 'embedding': embedding}},
                    upsert=True
                )
            except Exception as e:
                logging.error(f"Error storing document: {e}")

        logging.info("Documents ingested and stored successfully.")
        return contents

    except Exception as e:
        logging.error(f"Error ingesting documents: {e}")
        raise

if __name__ == "__main__":
    ingest_documents()
