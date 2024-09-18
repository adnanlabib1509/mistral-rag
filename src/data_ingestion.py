import os
from typing import List, Dict
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from src.config import DOCUMENTS_DIR
from src.utils import clean_text

def read_pdf_with_ocr(file_path: str) -> str:
    """
    Read content from a PDF file, using OCR if necessary.
    """
    # First, try to extract text directly
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        text = ' '.join(page.extract_text() for page in pdf.pages)
    
    # If no text was extracted, use OCR
    if not text.strip():
        images = convert_from_path(file_path)
        text = ' '.join(pytesseract.image_to_string(image) for image in images)
    
    return text

def load_documents() -> List[Dict[str, str]]:
    """
    Load documents from the documents directory.
    Returns a list of dictionaries containing document content and metadata.
    Handles both .txt and .pdf files, using OCR for PDFs if necessary.
    """
    documents = []
    for filename in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        if filename.endswith('.txt'):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        elif filename.endswith('.pdf'):
            content = read_pdf_with_ocr(file_path)
        else:
            continue  # Skip files that are neither .txt nor .pdf

        if content:  # Only add if content is not empty
            documents.append({
                "id": filename,
                "content": clean_text(content),
                "metadata": {"source": filename}
            })
    return documents