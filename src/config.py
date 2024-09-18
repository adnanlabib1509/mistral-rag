import os

# Model configurations
MISTRAL_MODEL = "mistralai/Mistral-7B-v0.3"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Data configurations
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

# Retriever configurations
TOP_K = 5
SIMILARITY_THRESHOLD = 0.7

# RAG configurations
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7