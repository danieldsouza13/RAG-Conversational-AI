from pymongo import MongoClient, ReadPreference
from pymongo.errors import ServerSelectionTimeoutError
from cohere import Client as CohereClient
from params import MONGODB_CONN_STRING, DB_NAME, COLLECTION_NAME, COHERE_API_KEY
from transformers import GPT2Tokenizer
from langchain.memory import ConversationBufferWindowMemory
import logging
import time

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Initialize memory buffer window to keep the last 5 conversations
memory = ConversationBufferWindowMemory(k=5)

def question_reshaping_decision(query, llm):
    logging.info("Step 1: Question Reshaping Decision")
    prompt = f"Does the following query need reshaping? Yes or No.\nQuery: {query}"
    response = llm.generate(prompt=prompt)
    decision = response.generations[0].text.strip().lower()
    needs_reshaping = 'yes' in decision
    logging.info(f"Question reshaping needed: {needs_reshaping}")
    return needs_reshaping

def standalone_question_generation(query, conversation_history, llm):
    logging.info("Step 2: Standalone Question Generation")
    context = "\n".join([f"User: {item['query']}\nAI: {item['answer']}" for item in conversation_history])
    prompt = f"{context}\nUser: {query}\nAI: Generate a standalone question that encapsulates all the required context."
    response = llm.generate(prompt=prompt)
    standalone_query = response.generations[0].text.strip()
    logging.info(f"Generated standalone question: {standalone_query}")
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

def run_rag_application(conversation_history):
    max_retries = 5
    backoff_time = 5

    try:
        llm = CohereClient(COHERE_API_KEY)

        query = input("Please enter your query: ")
        logging.info(f"User query: {query}")

        # Step 1: Question Reshaping Decision
        needs_reshaping = question_reshaping_decision(query, llm)
        if needs_reshaping:
            query = standalone_question_generation(query, conversation_history, llm)
            logging.info(f"Standalone question: {query}")

        for attempt in range(max_retries):
            try:
                logging.info(f"Attempting to connect to MongoDB (Attempt {attempt + 1})...")
                mongo_client = MongoClient(
                    MONGODB_CONN_STRING,
                    serverSelectionTimeoutMS=5000,
                    read_preference=ReadPreference.SECONDARY_PREFERRED
                )
                db = mongo_client[DB_NAME]
                collection = db[COLLECTION_NAME]
                mongo_client.admin.command('ping')
                logging.info("Successfully connected to MongoDB.")
                break
            except ServerSelectionTimeoutError as e:
                logging.error(f"MongoDB connection attempt {attempt + 1} failed: {e}")
                time.sleep(backoff_time)
                backoff_time *= 2
        else:
            raise Exception("Could not connect to MongoDB after several attempts.")

        logging.info("Loading documents from MongoDB...")
        documents = list(collection.find({}, {'_id': 0, 'content': 1, 'embedding': 1}))
        logging.info(f"Loaded {len(documents)} documents.")

        if not documents:
            raise ValueError("No documents found in the database. Please run the ingestion script first.")
        
        document_embeddings = [doc['embedding'] for doc in documents]

        route, docs = inner_router_decision(query, document_embeddings, documents, llm)
        if route == 'rag_app':
            logging.info("Step 4: RAG Application Route")
            if docs:
                context = "\n".join([doc.get('content', 'No content available') for doc in docs])
                truncated_context = truncate_context(context, query)
                prompt = f"Context: {truncated_context}\nQuery: {query}\nAI: "
                response = llm.generate(prompt=prompt)
            else:
                logging.info("No relevant documents found, returning 'no answer' response")
                response = llm.generate(prompt=query)
        else:
            logging.info("Step 4: Chat Model Route")
            context = "\n".join([f"User: {item['query']}\nAI: {item['answer']}" for item in conversation_history])
            truncated_context = truncate_context(context, query)
            prompt = f"{truncated_context}\nUser: {query}\nAI: "
            response = llm.generate(prompt=prompt)

        answer = response.generations[0].text.strip()
        logging.info(f"Generated answer: {answer}")

        # Update the memory with the latest conversation
        memory.save_context({"input": query}, {"output": answer})
        conversation_history = memory.load_memory_variables({})['history']

        return documents, answer, query

    except Exception as e:
        logging.error(f"Error in RAG application: {e}")
        raise
