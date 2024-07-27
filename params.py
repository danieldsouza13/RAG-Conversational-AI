from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MONGODB_CONN_STRING = os.getenv('MONGODB_CONN_STRING')
DB_NAME = os.getenv('DB_NAME')
DOCS_COLLECTION = os.getenv('DOCS_COLLECTION')
CHATLOG_COLLECTION = os.getenv('CHATLOG_COLLECTION')
EVALUATION_COLLECTION = os.getenv('EVALUATION_COLLECTION')

# Langsmith settings
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')
langchain_tracing_v2 = 'true'
langchain_endpoint = 'https://api.smith.langchain.com'