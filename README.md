# MistralRAG: Advanced Retrieval-Augmented Generation System

MistralRAG is a Retrieval-Augmented Generation (RAG) system that leverages Mistral 7B to provide accurate and contextually relevant responses to user queries. This project demonstrates the integration of various NLP techniques and models to create a comprehensive question-answering system.

## Features

- Document ingestion and preprocessing for both .txt and .pdf files
- OCR capability for scanned PDFs using Tesseract
- Efficient document embedding using a state-of-the-art sentence transformer model
- Fast and accurate document retrieval based on semantic similarity
- Advanced language generation using Mistral's 7B parameter model
- Modular and extensible architecture for easy customization and improvement

## How It Works

1. **Data Ingestion**: The system loads and preprocesses documents from a specified directory. It can handle both .txt and .pdf files, including scanned PDFs using OCR.

2. **Document Embedding**: Using a pre-trained sentence transformer model, the system creates dense vector representations (embeddings) for each document. These embeddings capture the semantic meaning of the documents.

3. **Query Processing**: When a user submits a query, it is embedded using the same sentence transformer model to ensure compatibility with the document embeddings.

4. **Document Retrieval**: The system uses cosine similarity to find the most relevant documents to the query. It retrieves the top-k documents that meet a specified similarity threshold.

5. **Context Preparation**: The retrieved documents are compiled into a context string, including their relevance scores.

6. **Prompt Creation**: A simple prompt is created by combining the user's query with the prepared context. This prompt is designed to guide the language model in generating a relevant response.

7. **Response Generation**: The Mistral 7B model processes the prompt and generates a response that aims to answer the user's query based on the provided context.

## Technical Details

- **Mistral Model**: We use the `mistralai/Mistral-7B-v0.3` model for text generation. This language model provides high-quality, coherent responses.

- **Embedding Model**: For document and query embedding, we use `sentence-transformers/all-MiniLM-L6-v2`, which provides a good balance between performance and efficiency.

- **Retrieval Mechanism**: We implement a custom retrieval system that uses cosine similarity to match query embeddings with document embeddings. The system is optimized for quick retrieval from a large set of documents.

- **RAG Implementation**: Our RAG system dynamically combines retrieved information with the power of the Mistral model, allowing for responses that are both informative and contextually appropriate.

- **PDF Processing**: We use PyPDF2 for text extraction from PDFs and Tesseract OCR (via pytesseract) for scanned documents.

## Project Structure

- `src/`: Contains the core components of the system.
  - `config.py`: Central configuration file for easy parameter tuning.
  - `data_ingestion.py`: Handles loading and preprocessing of documents.
  - `embeddings.py`: Manages the creation and storage of document embeddings.
  - `retriever.py`: Implements the document retrieval mechanism.
  - `model.py`: Wrapper for the Mistral language model.
  - `rag.py`: Orchestrates the entire RAG process.
  - `main.py`: The main file which simulates everything.

- `tests/`: Contains unit tests for each component to ensure reliability.
  - `test_data_ingestion.py`: Tests for document loading functionality.
  - `test_embeddings.py`: Tests for the embedding process.
  - `test_retriever.py`: Tests for the retrieval mechanism.
  - `test_model.py`: Tests for the Mistral model wrapper.
  - `test_rag.py`: Tests for the overall RAG system.

## Setup and Usage

1. Clone the repository:
```
git clone https://github.com/yourusername/mistral-rag.git
cd mistral-rag
```

2. Install the required dependencies:
```
pip install -e .
```
3. Install Tesseract OCR on your system (for PDF OCR capability).

4. Place your documents (.txt or .pdf) in the `data/documents/` directory.

5. Run the main script:
```
python src/main.py
```

This will start an interactive session where you can input queries and receive responses from the system. The script will:

1. Load all documents from the `data/documents/` directory
2. Initialize the RAG system
3. Prompt you for questions
4. For each question, it will:
   - Retrieve relevant documents
   - Generate a response using the Mistral model
   - Display the retrieved documents and the generated response

Type 'quit' to exit the interactive session.

## Running Tests

To run the test suite:
```
python -m unittest discover tests
```
## Future Improvements

- Implement streaming responses for faster user interaction
- Add support for multi-modal inputs (e.g., images, audio)
- Enhance the retrieval system with hybrid search (combining dense and sparse retrievers)
- Implement a caching mechanism for frequently asked queries
- Add a web interface for easier interaction with the system

## License

This project is licensed under the MIT License.

## Acknowledgments

- Mistral AI for their incredible language model
- The Hugging Face team for their transformers library
- The open-source NLP community for their continuous innovations
- The Tesseract OCR project for enabling text extraction from scanned documents