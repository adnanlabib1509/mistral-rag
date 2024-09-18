import unittest
from src.data_ingestion import load_documents
from src.config import DOCUMENTS_DIR
import os

class TestDataIngestion(unittest.TestCase):
    def setUp(self):
        # Create a test document
        self.test_doc_path = os.path.join(DOCUMENTS_DIR, "test_doc.txt")
        with open(self.test_doc_path, "w") as f:
            f.write("This is a test document.")

    def tearDown(self):
        # Remove the test document
        os.remove(self.test_doc_path)

    def test_load_documents(self):
        documents = load_documents()
        self.assertTrue(any(doc["id"] == "test_doc.txt" for doc in documents))
        self.assertTrue(any("test document" in doc["content"] for doc in documents))

if __name__ == '__main__':
    unittest.main()