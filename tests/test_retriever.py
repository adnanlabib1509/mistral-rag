import unittest
import numpy as np
from src.retriever import Retriever

class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = Retriever()

    def test_retrieve(self):
        query_embedding = np.random.rand(384)  # Assuming embedding size is 384
        retrieved_docs = self.retriever.retrieve(query_embedding)
        self.assertIsInstance(retrieved_docs, list)
        self.assertLessEqual(len(retrieved_docs), 5)  # As per TOP_K in config
        if retrieved_docs:
            self.assertIn('id', retrieved_docs[0])
            self.assertIn('similarity', retrieved_docs[0])

if __name__ == '__main__':
    unittest.main()