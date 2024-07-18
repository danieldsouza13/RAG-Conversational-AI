import os
import logging
from rag_app import run_rag_application, connect_to_mongodb
from eval_script import run_evaluation
import params
import uuid

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("\nWelcome to the RAG Conversational AI!\n")
    setup_logging()

    # Set USER_AGENT environment variable
    os.environ['USER_AGENT'] = 'my-custom-user-agent/1.0'

    # Establish a single MongoDB connection at the start
    db = connect_to_mongodb()

    conversation_id = str(uuid.uuid4())  # Generate a unique conversation ID for the session
    conversation_history = []  # List to store conversation history

    while True:
        try:
            logging.info("Running RAG Conversational AI...")
            documents, answer, query = run_rag_application(conversation_history, db, conversation_id)
            conversation_history.append({"query": query, "answer": answer})  # Store the conversation history
        except Exception as e:
            logging.error(f"Error during RAG Conversational AI run: {e}")
            continue

        choice = input("\nDo you want to (1) Ask another question, (2) Run the evaluation script, or (3) Exit? Enter 1, 2, or 3: ")
        print()
        if choice == "1":
            continue
        elif choice == "2":
            try:
                run_evaluation(documents, answer, query)
            except Exception as e:
                logging.error(f"Error during evaluation: {e}")
        elif choice == "3":
            break
        else:
            logging.warning("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
