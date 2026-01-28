"""
Embedding generation tests.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.embeddings import EmbeddingModel
from src.data.processor import DocumentProcessor

def test_embedding_dimensions():
    print("Testing embedding dimensions...")
    model = EmbeddingModel()
    texts = ["test sentence"]
    embeddings = model.encode(texts)
    print(f"Shape: {embeddings.shape}")
    assert embeddings.shape == (1, 384), f"Expected (1, 384), got {embeddings.shape}"
    print("✓ PASS")

def test_chunk_embeddings():
    print("Testing chunk embedding integration...")
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    model = EmbeddingModel()
    
    doc = {'content': 'This is a test document. ' * 50, 'metadata': {'source': 'test.txt'}}
    chunks = processor.process_document(doc)
    
    if not chunks:
        print("✗ FAIL: No chunks generated")
        return False
        
    chunk_texts = [chunk.content for chunk in chunks]
    embeddings = model.encode(chunk_texts)
    
    print(f"Generated embeddings for {len(chunks)} chunks, shape: {embeddings.shape}")
    assert embeddings.shape[0] == len(chunks), f"Expected {len(chunks)} embeddings, got {embeddings.shape[0]}"
    assert embeddings.shape[1] == 384, f"Expected 384 dimensions, got {embeddings.shape[1]}"
    print("✓ PASS")
    return True

if __name__ == '__main__':
    test_embedding_dimensions()
    test_chunk_embeddings()
    print("All embedding tests passed!")
