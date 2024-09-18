from src.data_ingestion import load_documents
from src.embeddings import DocumentEmbedder
from src.retriever import Retriever
from src.model import MistralModel
from typing import List, Dict

class RAGSystem:
    def __init__(self):
        self.documents = load_documents()
        self.embedder = DocumentEmbedder()
        self.retriever = Retriever()
        self.model = MistralModel()
        
        # Embed documents if not already done
        self.embedder.embed_documents(self.documents)
        
    def process_query(self, query: str) -> Dict[str, any]:
        """
        Process a query through the RAG system.
        """
        query_embedding = self.embedder.embed_query(query)
        retrieved_docs = self.retriever.retrieve(query_embedding)
        
        context = self._prepare_context(retrieved_docs)
        prompt = self._create_prompt(query, context)
        
        response = self.model.generate(prompt)
        
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "response": response
        }
    
    def _prepare_context(self, retrieved_docs: List[Dict[str, any]]) -> str:
        context = ""
        for doc in retrieved_docs:
            doc_content = next(d["content"] for d in self.documents if d["id"] == doc["id"])
            context += f"Document {doc['id']} (Similarity: {doc['similarity']:.2f}):\n{doc_content}\n\n"
        return context
    
    def _create_prompt(self, query: str, context: str) -> str:
        return f"""Context information is below.
                    ---------------------
                    {context}
                    ---------------------
                    Given the context information and not prior knowledge, answer the query.
                    Query: {query}
                    Answer:"""