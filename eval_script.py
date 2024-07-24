import json
import logging
from cohere import Client as CohereClient
from datasets import Dataset
from params import COHERE_API_KEY, EVALUATION_COLLECTION

# Initialize Cohere client
cohere_client = CohereClient(COHERE_API_KEY)

def load_test_set(test_set_path):
    with open(test_set_path, 'r') as file:
        test_set = json.load(file)
    return test_set

def prepare_dataset(test_set):
    queries = test_set['question']
    contexts = test_set['contexts']
    ground_truths = test_set['ground_truth']
    
    dataset_dict = {
        'question': queries,
        'contexts': contexts,
        'ground_truth': ground_truths,
    }
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

def embed_texts(texts):
    response = cohere_client.embed(texts=texts)
    return response.embeddings

def cosine_similarity(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2)) / ((sum(a**2 for a in vec1)**0.5) * (sum(b**2 for b in vec2)**0.5))

def context_precision(dataset):
    scores = []
    for data in dataset:
        question_embedding = embed_texts([data['question']])[0]
        context_embeddings = embed_texts(data['contexts'])
        max_similarity = max([cosine_similarity(question_embedding, context_embedding) for context_embedding in context_embeddings])
        scores.append(max_similarity)
    return sum(scores) / len(scores)

def context_recall(dataset):
    scores = []
    for data in dataset:
        ground_truth_embedding = embed_texts([data['ground_truth']])[0]
        context_embeddings = embed_texts(data['contexts'])
        max_similarity = max([cosine_similarity(ground_truth_embedding, context_embedding) for context_embedding in context_embeddings])
        scores.append(max_similarity)
    return sum(scores) / len(scores)

def evaluate_individual(data):
    question_embedding = embed_texts([data['question']])[0]
    context_embeddings = embed_texts(data['contexts'])
    ground_truth_embedding = embed_texts([data['ground_truth']])[0]
    
    context_scores = [cosine_similarity(question_embedding, context_embedding) for context_embedding in context_embeddings]
    context_precision_score = max(context_scores)
    
    ground_truth_scores = [cosine_similarity(ground_truth_embedding, context_embedding) for context_embedding in context_embeddings]
    context_recall_score = max(ground_truth_scores)
    
    return {
        'context_precision': context_precision_score,
        'context_recall': context_recall_score,
    }

def run_evaluation(dataset, db):
    try:
        collection = db[EVALUATION_COLLECTION]

        results = []
        for data in dataset:
            evaluation = evaluate_individual(data)
            result = {
                'question': data['question'],
                'ground_truths': data['ground_truth'],
                'contexts': data['contexts'],
                'context_precision': evaluation['context_precision'],
                'context_recall': evaluation['context_recall'],
            }
            results.append(result)
            collection.insert_one(result)
        logging.info("Evaluation completed successfully, results saved to database.")

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise
