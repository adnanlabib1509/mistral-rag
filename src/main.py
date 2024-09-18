# src/main.py

from src.data_ingestion import load_documents
from src.embeddings import DocumentEmbedder
from src.retriever import Retriever
from src.model import MistralModel
from src.rag import RAGSystem

def main():
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    print("Initializing RAG system...")
    rag_system = RAGSystem()

    print("RAG system ready. You can start asking questions. Type 'quit' to exit.")

    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'quit':
            break

        print("Processing query...")
        result = rag_system.process_query(query)

        print("\nRetrieved Documents:")
        for doc in result['retrieved_documents']:
            print(f"- {doc['id']} (Similarity: {doc['similarity']:.2f})")

        print("\nGenerated Response:")
        print(result['response'])

    print("Thank you for using MistralRAG!")

if __name__ == "__main__":
    main()