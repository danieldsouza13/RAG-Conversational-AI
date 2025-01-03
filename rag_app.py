from pymongo import MongoClient, ReadPreference
from pymongo.errors import ServerSelectionTimeoutError
from cohere import Client as CohereClient
from params import MONGODB_CONN_STRING, DB_NAME, DOCS_COLLECTION, CHATLOG_COLLECTION, COHERE_API_KEY
from transformers import GPT2Tokenizer
import logging
import time
import uuid
from datetime import datetime

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

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
    collection = db[DOCS_COLLECTION]
    documents = list(collection.find({}, {'_id': 0, 'content': 1, 'embedding': 1}))
    logging.info(f"Loaded {len(documents)} documents.")
    if not documents:
        raise ValueError("No documents found in the database. Please run the ingestion script first.")
    return documents

def question_reshaping_decision(query, conversation_history, llm):
    logging.info("Step 1: Question Reshaping Decision")
    prompt = f"You are part of a conversational AI system that determines whether to use a retrieval-augmented generator (RAG) or a chat model to answer user queries. Determine if the following user query needs reshaping according to chat history to provide necessary context and information for answering. Only respond with 'Yes' or 'No'.\nQuery: {query} \nChat History: {conversation_history}"
    response = llm.generate(prompt=prompt)
    decision = response.generations[0].text.strip().lower()
    needs_reshaping = 'Yes' in decision
    logging.info(f"Question reshaping needed: {needs_reshaping}")
    return needs_reshaping

# Generates a standalone question that encapsulates the user's query WITH the chat history.
def standalone_question_generation(query, conversation_history, llm):
    logging.info("Step 2: Standalone Question Generation")
    prompt = f"You are part of a conversational AI system that determines whether to use a retrieval-augmented generator (RAG) or a chat model to answer user queries. Take the original user query and chat history, and generate a new standalone question that can be understood and answered without relying on additional external information.\nQuery: {query} \nChat History: {conversation_history}"
    response = llm.generate(prompt=prompt)
    standalone_query = response.generations[0].text.strip()
    logging.info(f"Standalone question: {standalone_query}")
    return standalone_query

def inner_router_decision(query, document_embeddings, documents, llm):
    logging.info("Step 3: Inner Router Decision")
    query_embedding = llm.embed(texts=[query]).embeddings[0]
    
    def cosine_similarity(vec1, vec2):
        return sum(a*b for a, b in zip(vec1, vec2)) / (sum(a**2 for a in vec1)**0.5 * sum(b**2 for b in vec2)**0.5)

    similarities = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in document_embeddings]
    most_relevant_index = similarities.index(max(similarities))
    most_relevant_doc = documents[most_relevant_index]

    threshold = 0.5
    if similarities[most_relevant_index] > threshold:
        return 'rag_app', [most_relevant_doc]
    else:
        return 'chat_model', None

def truncate_context(context, query, max_tokens=4000):
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
    doc_context = "\n".join([doc.get('content', 'No content available') for doc in docs])
    trunucated_doc_context = truncate_context(doc_context, query)
    prompt = f"User Query: {query}\n Context: {trunucated_doc_context}"
    response = llm.generate(prompt=prompt)
    return response, doc_context

def handle_chat_model_route(query, conversation_history, llm):
    prompt = f"User Query: {query}\n Chat History: {conversation_history}"
    response = llm.generate(prompt=prompt)
    return response

def log_chat_to_mongodb(db, log_entry):
    try:
        collection = db[CHATLOG_COLLECTION]
        collection.insert_one(log_entry)
        logging.info("Chat log stored in MongoDB.")
    except Exception as e:
        logging.error(f"Error logging chat to MongoDB: {e}")

def run_rag_application(conversation_history, db, conversation_id):
    chat_id = str(uuid.uuid4())  # Generate a unique chat ID for each query
    start_time = time.time()  # Track the start time

    try:
        llm = CohereClient(COHERE_API_KEY)

        query = input("Please enter your query: ")
        logging.info(f"User query: {query}")

        # Step 1: Question Reshaping Decision
        needs_reshaping = question_reshaping_decision(query, conversation_history, llm)
        if needs_reshaping:
            query = standalone_question_generation(query, conversation_history, llm)

        documents = load_documents(db)
        document_embeddings = [doc['embedding'] for doc in documents]

        route, docs = inner_router_decision(query, document_embeddings, documents, llm)
        context = ""
        document_details = []
        
        # RAG Application Route for when a document is relevant to the user query and RAG is needed to generate an answer
        if route == 'rag_app':
            logging.info("Step 4a: RAG Application Route") 
            response, context = handle_rag_route(query, docs, llm)
            document_details = [{"content": doc.get('content', 'No content available')} for doc in docs]
        # Chat Model Route for when no documents are relevant to the user query and the chat model has sufficient knowledge to generate an answer.
        elif route == 'chat_model':
            logging.info("Step 4b: Chat Model Route")
            response = handle_chat_model_route(query, conversation_history, llm)
            context = None
            document_details = None

        answer = response.generations[0].text.strip()
        logging.info(f"Generated answer: {answer}")

        # Update the memory with the latest conversation
        conversation_history.append(["human", query])
        conversation_history.append(["ai", answer])

        # Keep only the last 5 conversations in history
        if len(conversation_history) > 10:  # 10 entries (5 queries and 5 answers)
            conversation_history = conversation_history[-10:]

        end_time = time.time()  # Track the end time
        response_time = end_time - start_time

        # Format and store the chat log into MongoDB database
        chat_log = {
            "query": query,
            "answer": answer,
            "chat_id": chat_id,
            "conversation_id": conversation_id,
            "route": route,
            "conversation_history": conversation_history,
            "model_name": "cohere",
            "total_tokens": len(tokenizer.encode(query)) + len(tokenizer.encode(answer)),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "response_time": response_time
        }
        if route == 'rag_app':
            chat_log["documents_used"] = document_details
            chat_log["context"] = context


        log_chat_to_mongodb(db, chat_log)

        return documents, answer, query

    except Exception as e:
        logging.error(f"Error in RAG application: {e}")
        raise