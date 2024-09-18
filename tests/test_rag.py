import unittest
from src.rag import RAGSystem

class TestRAG(unittest.TestCase):
    def setUp(self):
        self.rag_system = RAGSystem()

    def test_process_query(self):
        query = "What is machine learning?"
        result = self.rag_system.process_query(query)
        self.assertIn('query', result)
        self.assertIn('retrieved_documents', result)
        self.assertIn('response', result)
        self.assertEqual(result['query'], query)
        self.assertIsInstance(result['retrieved_documents'], list)
        self.assertIsInstance(result['response'], str)

if __name__ == '__main__':
    unittest.main()