from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from datasets import Dataset
import logging

def run_evaluation(dataset: Dataset):
    try:
        # Initialize your metrics
        metrics = [
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ]

        # Run the evaluation
        result = evaluate(dataset, metrics=metrics)

        # Print the result
        logging.info(result)

        # Convert the result to a pandas DataFrame for further analysis
        df = result.to_pandas()
        logging.info(df.head())

        return df

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise
