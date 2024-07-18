from pymongo import MongoClient, ReadPreference
from pymongo.errors import ServerSelectionTimeoutError
from cohere import Client as CohereClient
from params import MONGODB_CONN_STRING, DB_NAME, COLLECTION_NAME, COHERE_API_KEY
from transformers import GPT2Tokenizer
from langchain.memory import ConversationBufferWindowMemory
import logging
import time

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Initialize sliding window memory to keep the last 5 conversations
memory = ConversationBufferWindowMemory(k=5)

def connect_to_mongodb():
    max_retries = 5
    backoff_time = 5
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempting to connect to MongoDB (Attempt {attempt + 1})...")
            mongo_client = MongoClient(
                MONGODB_CONN_STRING,
                serverSelectionTimeoutMS=5000,
                read_preference=ReadPreference.SECONDARY_PREFERRED
            )
            db = mongo_client[DB_NAME]
            mongo_client.admin.command('ping')
            logging.info("Successfully connected to MongoDB.")
            return db
        except ServerSelectionTimeoutError as e:
            logging.error(f"MongoDB connection attempt {attempt + 1} failed: {e}")
            time.sleep(backoff_time)
            backoff_time *= 2
    raise Exception("Could not connect to MongoDB after several attempts.")

def load_documents(db):
    logging.info("Loading documents from MongoDB...")
    collection = db[COLLECTION_NAME]
    documents = list(collection.find({}, {'_id': 0, 'content': 1, 'embedding': 1}))
    logging.info(f"Loaded {len(documents)} documents.")
    if not documents:
        raise ValueError("No documents found in the database. Please run the ingestion script first.")
    return documents

def question_reshaping_decision(query, conversation_history, llm):
    logging.info("Step 1: Question Reshaping Decision")
    structured_conversation_history = "\n".join([f"User: {item['query']}\nAI: {item['answer']}\n" for item in conversation_history])
    prompt = f"Determine if the following user query needs reshaping according to chat history to provide necessary context and information for answering. Respond with 'Yes' or 'No'.\nQuery: {query} \nChat History: {structured_conversation_history}"
    response = llm.generate(prompt=prompt)
    decision = response.generations[0].text.strip().lower()
    needs_reshaping = 'yes' in decision
    logging.info(f"Question reshaping needed: {needs_reshaping}")
    return needs_reshaping

# Generates a standalone question that encapsulates the user's query WITH the chat history.
def standalone_question_generation(query, conversation_history, llm):
    logging.info("Step 2: Standalone Question Generation")
    structured_conversation_history = "\n".join([f"User: {item['query']}\nAI: {item['answer']}\n" for item in conversation_history])
    prompt = f"Take the original user query and chat history, and generate a new standalone question that can be understood and answered without relying on additional external information.\nQuery: {query} \nChat History: {structured_conversation_history}"
    response = llm.generate(prompt=prompt)
    standalone_query = response.generations[0].text.strip()
    logging.info(f"Standalone question: {standalone_query}")
    return standalone_query

def inner_router_decision(query, document_embeddings, documents, llm):
    # Embeds the query and calculates the cosine similarity with document embeddings to find the most relevant document.
    logging.info("Step 3: Inner Router Decision")
    query_embedding = llm.embed(texts=[query]).embeddings[0]
    
    def cosine_similarity(vec1, vec2):
        return sum(a*b for a, b in zip(vec1, vec2)) / (sum(a**2 for a in vec1)**0.5 * sum(b**2 for b in vec2)**0.5)

    similarities = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in document_embeddings]
    most_relevant_index = similarities.index(max(similarities))
    most_relevant_doc = documents[most_relevant_index]

    threshold = 0.5
    # If the most relevant document has a similarity score above the threshold, return the RAG Application route. Otherwise return the Chat Model route.
    if similarities[most_relevant_index] > threshold:
        return 'rag_app', [most_relevant_doc]
    else:
        prompt = f"Query: {query}\nDetermine the best path for obtaining a comprehensive answer."
        response = llm.generate(prompt=prompt)
        decision = response.generations[0].text.strip().lower()
        if 'rag application' in decision:
            return 'rag_app', None
        else:
            return 'chat_model', None

def truncate_context(context, query, max_tokens=4000):
    # Tokenize the context and query
    query_tokens = tokenizer.encode(query)
    context_tokens = tokenizer.encode(context)
    
    prompt_structure_tokens = 100  # Approximate number of tokens for "Context: ", "Query: ", and other fixed parts
    available_tokens = max_tokens - len(query_tokens) - prompt_structure_tokens

    if len(context_tokens) > available_tokens:
        truncated_context_tokens = context_tokens[-available_tokens:]
        truncated_context = tokenizer.decode(truncated_context_tokens)
        logging.warning(f"Context truncated to the last {available_tokens} tokens.")
        return truncated_context
    return context

def handle_rag_route(query, docs, llm):
    context = "\n".join([doc.get('content', 'No content available') for doc in docs])
    truncated_context = truncate_context(context, query)
    prompt = f"Context: {truncated_context}\nQuery: {query}\nAI: "
    response = llm.generate(prompt=prompt)
    return response

def handle_chat_model_route(query, conversation_history, llm):
    context = "\n".join([f"User: {item['query']}\nAI: {item['answer']}" for item in conversation_history])
    truncated_context = truncate_context(context, query)
    prompt = f"{truncated_context}\nUser: {query}\nAI: "
    response = llm.generate(prompt=prompt)
    return response

def run_rag_application(conversation_history):
    try:
        llm = CohereClient(COHERE_API_KEY)

        query = input("Please enter your query: ")
        logging.info(f"User query: {query}")

        # Step 1: Question Reshaping Decision
        needs_reshaping = question_reshaping_decision(query, conversation_history, llm)
        if needs_reshaping:
            query = standalone_question_generation(query, conversation_history, llm)

        db = connect_to_mongodb()
        documents = load_documents(db)
        document_embeddings = [doc['embedding'] for doc in documents]

        route, docs = inner_router_decision(query, document_embeddings, documents, llm)
        if route == 'rag_app':
            logging.info("Step 4a: RAG Application Route")
            response = handle_rag_route(query, docs, llm)

        else:
            logging.info("Step 4b: Chat Model Route")
            response = handle_chat_model_route(query, conversation_history, llm)

        answer = response.generations[0].text.strip()
        logging.info(f"Generated answer: {answer}")

        # Update the memory with the latest conversation
        memory.save_context({"input": query}, {"output": answer})
        conversation_history = memory.load_memory_variables({})['history']

        return documents, answer, query

    except Exception as e:
        logging.error(f"Error in RAG application: {e}")
        raise
