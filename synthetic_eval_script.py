import logging
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, faithfulness, context_relevancy, answer_relevancy
from params import TESTSET_COLLECTION, INITIAL_BENCHMARK_COLLECTION, OPENAI_API_KEY
from datasets import Dataset
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pymongo import InsertOne

def run_evaluation(db):
    try:
        # Initialize the specific OpenAI models
        llm = ChatOpenAI(model="gpt-4o-mini")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Initialize your metrics
        metrics = [
            context_recall,
            context_precision,
            faithfulness,
            context_relevancy,
            answer_relevancy
        ]

        # Retrieve the testset from MongoDB
        testsets_collection = db[TESTSET_COLLECTION]
        # When retrieving the testset, sort by the original insertion order / sequence number
        testset_data = list(testsets_collection.find({}, {'_id': 0}))

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(testset_data)

        # Ensure required columns exist
        required_columns = ['sequence_number', 'question', 'answer', 'contexts', 'ground_truth', 'question_type', 'episode_done', 'response_time']
        for column in required_columns:
            if column not in dataset.column_names:
                raise ValueError(f"Required column '{column}' not found in the dataset")

        # Evaluate the dataset
        results = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)
        dataset.sort("sequence_number")

        # Prepare documents for MongoDB
        documents = []
        for i in range(len(dataset)):
            document = {
                "sequence_number": dataset[i]["sequence_number"],
                "question": dataset[i]["question"],
                "answer": dataset[i]["answer"],
                "contexts": dataset[i]["contexts"],
                "ground_truth": dataset[i]["ground_truth"],
                "context_precision": results.scores["context_precision"][i],
                "context_recall": results.scores["context_recall"][i],
                "faithfulness": results.scores["faithfulness"][i],
                "context_relevancy": results.scores["context_relevancy"][i],
                "answer_relevancy": results.scores["answer_relevancy"][i],
                "question_type": dataset[i]["question_type"],
                "episode_done": dataset[i]["episode_done"],
                "response_time": dataset[i]["response_time"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            documents.append(InsertOne(document))

         # Insert documents into MongoDB using bulk write with ordered=True
        result = db[INITIAL_BENCHMARK_COLLECTION].bulk_write(documents, ordered=True)
        logging.info(f"{result.inserted_count} evaluation results logged to {INITIAL_BENCHMARK_COLLECTION}")

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise