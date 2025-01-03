# RAG Conversational AI and Evaluation Framework

This project combines a **Retrieval-Augmented Generation (RAG)** Conversational AI system with a robust **Evaluation Framework**. The RAG system integrates Cohere's LLM, MongoDB for storage, and LangChain for intelligent query processing, while the Evaluation Framework enables benchmarking with synthetic test sets, performance metrics, and detailed visualizations.

---

## Features

### Conversational AI
- **Document Ingestion**: Processes JSON files, generates embeddings using Cohere, and stores them in MongoDB.
- **Dynamic Query Routing**: Automatically decides whether to use RAG or a pure chat model based on query relevance.
- **Context Management**: Maintains conversation history and handles token-aware context truncation.
- **Error Handling**: Implements retry mechanisms for MongoDB connections and logs errors comprehensively.

### Evaluation Framework
- **Synthetic Testset Generation**: Creates test queries with ground truths using GPT-based models.
- **Performance Metrics**: Evaluates precision, recall, faithfulness, relevancy, and response times using RAGAS.
- **Visualization**: Generates radar charts, histograms, confusion matrices, and execution time distributions.
- **PDF Reporting**: Produces detailed performance reports with improvement suggestions.

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- MongoDB Atlas account
- Cohere API key

### 1. Clone the Repository
    git clone <your_github_repo_url>
    cd RAG-Conversational-AI

### 2. Install Dependencies
    pip install -r requirements.txt

### 3. Create a `.env` File
Create a `.env` file in the root directory of your project with following the setup in `.env.example`

### 4. Ingest Documents into MongoDB
Run the document ingestion script to add sample documents to your MongoDB database:
    python ingest_docs.py

### 5. Start the RAG Conversational AI Application
Run the Conversational AI:
    python main.py

## Eval Framework Instructions

### 1. Generate Synthetic Testset
Use the `synthetic_testset.py` script to create test queries with ground truths:
    python synthetic_testset.py

### 2. Run Evaluation
Evaluate the system's performance using various metrics:
    python synthetic_eval_script.py
The script will calculate metrics such as precision, recall, faithfulness, relevancy, cost, and response time.

### 3. Generate Benchmarking PDF Report
Create a detailed performance report with visualizations:
    python evaluation_report.py
The report includes radar charts, histograms, confusion matrices, and improvement suggestions.

---

## Usage

1. **Ask Questions**: Enter your query when prompted. The system will:
   - Reshape the question if needed.
   - Decide whether to use RAG (retrieval-based) or Chat (LLM-only) mode.
   - Generate a response based on the selected route.

2. **Run Evaluations** Select the evaluation option from the menu to test performance metrics.

3. **Exit**: Choose the exit option when you're done.

---

## System Architecture

### Document Ingestion Pipeline:
- Reads JSONL documents.
- Generates embeddings using Cohere's API.
- Stores content and embeddings in MongoDB.

### Query Processing Flow:
1. **Question Reshaping Decision**: Determines if the query needs additional context from conversation history.
2. **Standalone Question Generation**: Reformulates queries into standalone questions if necessary.
3. **Inner Router Decision**: Uses cosine similarity to decide between RAG or Chat mode.
4. **Response Generation**:
   - **RAG Mode**: Retrieves relevant documents and generates responses based on them.
   - **Chat Mode**: Generates responses directly using Cohere's LLM.

### Performance Metrics
The evaluation framework uses the following metrics:
- **Context Precision**: Measures how much of the retrieved context is relevant.
- **Context Recall**: Measures how much relevant information is retrieved.
- **Faithfulness**: Measures how well the answer aligns with the provided context.
- **Context Relevancy**: Measures how relevant the retrieved context is to the question.
- **Answer Relevancy**: Measures how relevant the generated answer is to the question.

### Visualization & Reporting
The evaluation framework provides:
1. Radar charts for median scores across key metrics.
2. Histograms for metric distributions (e.g., precision, recall).
3. Confusion matrices showing retrieval and answer accuracy rates.
4. Execution time distribution histograms for performance analysis.
5. PDF reports summarizing all results with actionable insights.

---

## Technical Details

### MongoDB Integration:
- Stores both documents and chat logs in separate collections.
- Implements retry mechanisms with exponential backoff for connection stability.

### Performance Features:
- Token-aware context truncation ensures queries stay within token limits.
- Conversation history is maintained for up to five previous interactions.

### Error Handling:
- Comprehensive logging system tracks all operations.
- Graceful degradation ensures smooth operation even during failures.

---

## Dependencies

The following Python libraries are required:
- `cohere`: For embeddings and language model interaction.
- `pymongo`: For MongoDB integration.
- `transformers`: For tokenization and context management.
- `ragas`: For evaluation metrics and testset generation.
- `reportlab`: For generating PDF reports.
- `matplotlib` & `pandas`: For data visualization and analysis.

Install them via `pip install -r requirements.txt`.

---

## Example Workflow

1. User enters a query: *"What is the capital of France?"*
2. The system checks if reshaping is needed (e.g., based on prior context).
3. If relevant documents are found in MongoDB, it uses RAG mode to generate an answer like *"The capital of France is Paris."*
4. If no relevant documents are found, it switches to Chat mode and generates an answer using Cohere's LLM.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---