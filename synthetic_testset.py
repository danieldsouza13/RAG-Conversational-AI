import os
import time
from pymongo import MongoClient, InsertOne
from params import MONGODB_CONN_STRING, DB_NAME, TESTSET_COLLECTION, OPENAI_API_KEY
from langchain_community.document_loaders import JSONLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datetime import datetime
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Function to get answer from the system
def get_answer(query:str, session_id:str="Ddsouza:test-session"):
    function_key = os.getenv('AZURE_FUNCTION_APP_KEY')
    url = os.getenv('AZURE_FUNCTION_APP_URL')+'customerservice/'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'x-functions-key': function_key
    }
    data = {
        'query': query,
        'session_id': session_id
    }
    response = requests.post(url, headers=headers, json=data)
    status_code = response.status_code

    if status_code == 200:
        response = response.json()
        response['status_code'] = status_code
        return response

    else:
        try:
            detail = response.json()['detail']
        except:
            detail = "Internal Server Error. Please try again later."
        resp = {
            'status_code':status_code,
            'detail': detail
        }
        return resp

# MongoDB setup
os.environ['USER_AGENT'] = "Daniel's-user-agent/1.0"
mongo_client = MongoClient(MONGODB_CONN_STRING)
db = mongo_client[DB_NAME]
collection = db[TESTSET_COLLECTION]

# JSONLoader configuration
loader = JSONLoader(
    file_path='Documents/prod_documents.json',
    jq_schema='.[]',
    content_key="textContent"
)

documents = loader.load()

for document in documents:
    document.metadata['filename'] = document.metadata['source']

# Generator with openai models
generator_llm = ChatOpenAI(model="gpt-4o-mini")
critic_llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# Generate testset
testset = generator.generate_with_langchain_docs(
    documents, 
    test_size=50, 
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}
)

# Convert testset to a list of dictionaries for MongoDB storage
testset_data = []
for index, row in testset.to_pandas().iterrows():
    testset_item = {
        "sequence_number": index + 1, 
        "question": row['question'],
        "answer": "",  
        "ground_truth": row['ground_truth'],
        "contexts": row['contexts'],
        "question_type": row['evolution_type'],
        "episode_done": row['episode_done'],
        "response_time": 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    testset_data.append(testset_item)

# Get answers and response times from the system
for index, item in enumerate(testset_data):
    print(f"\n{index + 1}. Question: {item['question']}")
    
    # Get answer from the system
    response = get_answer(query=item['question'], session_id=f"test_session-synthetic_testset_{index + 1}")
    
    if response['status_code'] == 200:
        item['answer'] = response['answer']
        item['response_time'] = response['total_time']
        print(f"Answer: {item['answer']}")
        print(f"Response time: {item['response_time']:.2f} seconds")
    else:
        print(f"Error: {response['detail']}")
        item['answer'] = "Error in retrieving answer"
        item['response_time'] = 0

# Prepare bulk write operations
bulk_operations = [InsertOne(item) for item in testset_data]

# Execute bulk write with ordered=True to maintain insertion order
result = collection.bulk_write(bulk_operations, ordered=True)

print(f"Inserted {result.inserted_count} documents into {TESTSET_COLLECTION}\n")