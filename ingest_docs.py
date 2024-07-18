from langchain_community.document_loaders import WebBaseLoader
from cohere import Client as CohereClient
from pymongo import MongoClient, ASCENDING
from params import COHERE_API_KEY, MONGODB_CONN_STRING, DB_NAME, DOCS_COLLECTION
import logging

def ingest_documents():
    try:
        # Load documents from the web
        urls = [ 'https://www.oracle.com/cloud/what-is-cloud-computing/#:~:text=In%20simple%20terms%2C%20cloud%20computing,it%20as%20they%20use%20it.',
            'https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-is-azure',
            'https://aws.amazon.com/what-is-aws/',
            'https://cloud.google.com/docs/overview',
            'https://learn.microsoft.com/en-us/azure/virtual-machines/overview',
            'https://cloud.google.com/kubernetes-engine/docs/concepts/kubernetes-engine-overview',
            'https://aws.amazon.com/pm/serv-s3/?gclid=Cj0KCQjwhb60BhClARIsABGGtw_51ynGbnMEtOMNaqTpMTPJC3tFEJgQVpGXa5M_IjNx0SObh9Mv7OcaAtY4EALw_wcB&trk=fecf68c9-3874-4ae2-a7ed-72b6d19c8034&sc_channel=ps&ef_id=Cj0KCQjwhb60BhClARIsABGGtw_51ynGbnMEtOMNaqTpMTPJC3tFEJgQVpGXa5M_IjNx0SObh9Mv7OcaAtY4EALw_wcB:G:s&s_kwcid=AL!4422!3!536456034896!p!!g!!aws%20s3%20cloud%20storage!11204620052!112938566794',
            'https://www.7mileadvisors.com/Whitepaper/advancements-of-cloud-technology/',
            'https://cloud.google.com/learn/advantages-of-cloud-computing',
            'https://cloud.google.com/learn/artificial-intelligence-vs-machine-learning?hl=en',]
        
        # For testing purposes I have chosen the documents to focus on the single topic of cloud computing and multiple topics of technology, Finance, and Healthcare.
        # Single Topic tests the AI's depth of understanding and ability to retrieve and generate responses from a concenrtated knowledge base, great for use as a tech support assistant for a specific software product.
        # Multiple Topics tests the AI's breadth of knowledge and ability to distinguish and accurately retieve relevant information from diverse subjcets, great for use as a gneeral knowledge assistant or customer service bot.

        loader = WebBaseLoader(urls)
        documents = loader.load()

        # Extract content from documents
        contents = [doc.page_content for doc in documents]

        # Initialize Cohere client for embeddings
        cohere_client = CohereClient(COHERE_API_KEY)

        # Generate embeddings for document contents
        embeddings = cohere_client.embed(texts=contents).embeddings

        # Initialize MongoDB client
        mongo_client = MongoClient(MONGODB_CONN_STRING)
        db = mongo_client[DB_NAME]
        collection = db[DOCS_COLLECTION]

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
