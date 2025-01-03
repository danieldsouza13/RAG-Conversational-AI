from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MONGODB_CONN_STRING = os.getenv('MONGODB_CONN_STRING')
DB_NAME = os.getenv('DB_NAME')
DOCS_COLLECTION = os.getenv('DOCS_COLLECTION')
CHATLOG_COLLECTION = os.getenv('CHATLOG_COLLECTION')
INITIAL_BENCHMARK_COLLECTION = os.getenv('INIITIAL_BENCHMARK_COLLECTION')
TESTSET_COLLECTION = os.getenv('TESTSET_COLLECTION')
INITIAL_BENCHMARK_COLLECTION = os.getenv('INITIAL_BENCHMARK_COLLECTION')
CONTINUOUS_MONITORING_COLLECTION = os.getenv('CONTINUOUS_MONITORING_COLLECTION')

# Langsmith environment variables
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')
langchain_tracing_v2 = 'true'
langchain_endpoint = 'https://api.smith.langchain.com'