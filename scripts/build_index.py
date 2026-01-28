"""
Script to build vector and BM25 indices from downloaded papers.
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DocumentLoader
from src.data.processor import DocumentProcessor
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.logger import log
from src.config import settings


def main():
    parser = argparse.ArgumentParser(description="Build RAG indices")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing documents to index"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for document processing"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing indices before building"
    )
    
    args = parser.parse_args()
    
    log.info("="*60)
    log.info("RAG Index Builder")
    log.info("="*60)
    
    # Set data directory
    data_dir = args.data_dir or "./data/raw"
    log.info(f"Data directory: {data_dir}")
    
    # Initialize components
    log.info("\nInitializing components...")
    loader = DocumentLoader()
    processor = DocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    vector_store = VectorStore()
    bm25_retriever = BM25Retriever()
    
    # Reset if requested
    if args.reset:
        log.info("Resetting existing indices...")
        vector_store.reset_collection()
    
    # Load documents
    log.info("\nLoading documents...")
    try:
        documents = loader.load_directory(
            directory=data_dir,
            recursive=True,
            file_types=[".pdf", ".txt", ".md"]
        )
        
        if not documents:
            log.error("No documents found to index!")
            sys.exit(1)
        
        doc_stats = loader.get_document_stats(documents)
        log.info(f"Loaded {doc_stats['total_documents']} documents")
        log.info(f"Total characters: {doc_stats['total_characters']:,}")
        log.info(f"File types: {doc_stats['file_types']}")
        
    except Exception as e:
        log.error(f"Error loading documents: {e}")
        sys.exit(1)
    
    # Process documents into chunks
    log.info("\nProcessing documents into chunks...")
    try:
        chunks = processor.process_documents(documents)
        
        if not chunks:
            log.error("No chunks created!")
            sys.exit(1)
        
        chunk_stats = processor.get_chunk_stats(chunks)
        log.info(f"Created {chunk_stats['total_chunks']} chunks")
        log.info(f"Average chunk size: {chunk_stats['avg_chunk_size_tokens']:.1f} tokens ({chunk_stats['avg_chunk_size_chars']:.0f} characters)")
        log.info(f"Token range: {chunk_stats['min_chunk_size_tokens']}-{chunk_stats['max_chunk_size_tokens']} tokens")
        log.info(f"Character range: {chunk_stats['min_chunk_size_chars']}-{chunk_stats['max_chunk_size_chars']} characters")
        log.info(f"Target: {chunk_stats['target_chunk_size_tokens']} tokens with {chunk_stats['target_overlap_tokens']} token overlap")
        
    except Exception as e:
        log.error(f"Error processing documents: {e}")
        sys.exit(1)
    
    # Build vector index
    log.info("\nBuilding vector index (this may take a while)...")
    try:
        vector_store.add_chunks(chunks, show_progress=True)
        
        vector_stats = vector_store.get_stats()
        log.info(f"Vector index built successfully")
        log.info(f"Total documents in index: {vector_stats['total_documents']}")
        log.info(f"Embedding dimension: {vector_stats['embedding_dimension']}")
        
    except Exception as e:
        log.error(f"Error building vector index: {e}")
        sys.exit(1)
    
    # Build BM25 index
    log.info("\nBuilding BM25 index...")
    try:
        bm25_retriever.build_index(chunks)
        bm25_retriever.save_index()
        
        bm25_stats = bm25_retriever.get_stats()
        log.info(f"BM25 index built successfully")
        log.info(f"Total documents in index: {bm25_stats['total_documents']}")
        log.info(f"Average document length: {bm25_stats['avg_doc_length']:.0f} words")
        
    except Exception as e:
        log.error(f"Error building BM25 index: {e}")
        sys.exit(1)
    
    # Summary
    log.info("\n" + "="*60)
    log.info("Index Building Complete!")
    log.info("="*60)
    log.info(f"✓ Vector index: {vector_stats['total_documents']} documents")
    log.info(f"✓ BM25 index: {bm25_stats['total_documents']} documents")
    log.info(f"✓ Total chunks: {chunk_stats['total_chunks']}")
    log.info("\nYou can now run the API server or query the system!")
    log.info("Run: python -m uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()
