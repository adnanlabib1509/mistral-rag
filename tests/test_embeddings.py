import unittest
import numpy as np
from src.embeddings import DocumentEmbedder

class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.embedder = DocumentEmbedder()

    def test_embed_query(self):
        query = "This is a test query"
        embedding = self.embedder.embed_query(query)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (384,))  # Assuming the embedding size is 384

if __name__ == '__main__':
    unittest.main()