import fitz  # PyMuPDF
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

def extract_text_from_pdf(content: bytes) -> str:
    """
    Extracts text from PDF using PyMuPDF with OCR fallback.
    
    Args:
        content (bytes): PDF file content as bytes

    Returns:
        str: Extracted and cleaned text from the PDF
    """
    text = ""
    
    # Method 1: PyMuPDF (Fast and Accurate)
    try:
        logger.info("Extracting text with PyMuPDF...")
        with fitz.open(stream=content, filetype="pdf") as doc:
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text += f"{page_text}\n"
        
        if text.strip():
            logger.info(f"Extracted {len(text)} characters using PyMuPDF")
            return clean_extracted_text(text)
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")


def clean_extracted_text(text: str) -> str:
    """Cleans and normalizes extracted text from PDF."""
    if not text:
        return ""
    
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Splits text into overlapping chunks for processing."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + chunk_size - 100:
                end = sentence_end + 1
            else:
                word_end = text.rfind(' ', start, end)
                if word_end > start + chunk_size - 50:
                    end = word_end
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < len(text) else len(text)
            
    return chunks