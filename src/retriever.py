import numpy as np
from typing import List, Dict
from src.config import EMBEDDINGS_DIR, TOP_K, SIMILARITY_THRESHOLD
import os

class Retriever:
    def __init__(self):
        self.document_embeddings = self._load_embeddings()
        
    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        embeddings = {}
        for filename in os.listdir(EMBEDDINGS_DIR):
            if filename.endswith(".npy"):
                embedding_path = os.path.join(EMBEDDINGS_DIR, filename)
                embeddings[filename[:-4]] = np.load(embedding_path)
        return embeddings
    
    def retrieve(self, query_embedding: np.ndarray) -> List[Dict[str, str]]:
        """
        Retrieve top-k most similar documents based on cosine similarity.
        """
        similarities = {}
        for doc_id, doc_embedding in self.document_embeddings.items():
            similarity = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
            similarities[doc_id] = similarity
        
        sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        retrieved_docs = []
        for doc_id, similarity in sorted_docs[:TOP_K]:
            if similarity >= SIMILARITY_THRESHOLD:
                retrieved_docs.append({"id": doc_id, "similarity": similarity})
        
        return retrieved_docs