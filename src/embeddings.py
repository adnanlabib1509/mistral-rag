import torch
from transformers import AutoTokenizer, AutoModel
from src.config import EMBEDDING_MODEL, EMBEDDINGS_DIR
import os

class DocumentEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL)
        
    def embed_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        Embed documents and save embeddings to disk.
        """
        for doc in documents:
            inputs = self.tokenizer(doc["content"], return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            # Save embeddings
            embedding_path = os.path.join(EMBEDDINGS_DIR, f"{doc['id']}.npy")
            np.save(embedding_path, embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string.
        """
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()