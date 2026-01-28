"""
Simple test script to verify the RAG system is working.
Run this after building the indices to test the system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.generation.llm import LLMClient
from src.generation.chain import RAGChain
from src.utils.logger import log


def test_components():
    """Test individual components."""
    print("\n" + "="*60)
    print("Testing RAG System Components")
    print("="*60)
    
    # Test 1: Vector Store
    print("\n[1/5] Testing Vector Store...")
    try:
        vector_store = VectorStore()
        stats = vector_store.get_stats()
        print(f"✓ Vector Store OK - {stats['total_documents']} documents indexed")
        if stats['total_documents'] == 0:
            print("⚠ Warning: No documents in vector store. Run build_index.py first.")
            return False
    except Exception as e:
        print(f"✗ Vector Store Error: {e}")
        return False
    
    # Test 2: BM25 Retriever
    print("\n[2/5] Testing BM25 Retriever...")
    try:
        bm25_retriever = BM25Retriever()
        if not bm25_retriever.load_index():
            print("⚠ Warning: BM25 index not found. Run build_index.py first.")
            return False
        stats = bm25_retriever.get_stats()
        print(f"✓ BM25 Retriever OK - {stats['total_documents']} documents indexed")
    except Exception as e:
        print(f"✗ BM25 Retriever Error: {e}")
        return False
    
    # Test 3: Reranker
    print("\n[3/5] Testing Reranker...")
    try:
        reranker = Reranker()
        print(f"✓ Reranker OK - Model: {reranker.model_name}")
    except Exception as e:
        print(f"✗ Reranker Error: {e}")
        return False
    
    # Test 4: LLM Client
    print("\n[4/5] Testing LLM Client...")
    try:
        llm_client = LLMClient()
        if not llm_client.check_model_availability():
            print(f"⚠ Warning: Model '{llm_client.model_name}' not available.")
            print(f"   Run: ollama pull {llm_client.model_name}")
            return False
        print(f"✓ LLM Client OK - Model: {llm_client.model_name}")
    except Exception as e:
        print(f"✗ LLM Client Error: {e}")
        print("   Make sure Ollama is installed and running.")
        return False
    
    # Test 5: RAG Chain
    print("\n[5/5] Testing RAG Chain...")
    try:
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            reranker=reranker
        )
        rag_chain = RAGChain(
            retriever=hybrid_retriever,
            llm_client=llm_client
        )
        print("✓ RAG Chain OK - All components initialized")
    except Exception as e:
        print(f"✗ RAG Chain Error: {e}")
        return False
    
    return True


def test_query():
    """Test a sample query."""
    print("\n" + "="*60)
    print("Testing Sample Query")
    print("="*60)
    
    try:
        # Initialize RAG chain
        print("\nInitializing RAG system...")
        vector_store = VectorStore()
        bm25_retriever = BM25Retriever()
        bm25_retriever.load_index()
        reranker = Reranker()
        
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            reranker=reranker
        )
        
        llm_client = LLMClient()
        rag_chain = RAGChain(
            retriever=hybrid_retriever,
            llm_client=llm_client
        )
        
        # Test query
        test_question = "What are transformers in machine learning?"
        print(f"\nQuery: {test_question}")
        print("\nRetrieving relevant documents...")
        
        result = rag_chain.query(
            query=test_question,
            top_k=3,
            use_reranking=True
        )
        
        print("\n" + "-"*60)
        print("ANSWER:")
        print("-"*60)
        print(result['answer'])
        
        print("\n" + "-"*60)
        print(f"SOURCES ({len(result['sources'])}):")
        print("-"*60)
        for source in result['sources']:
            print(f"\n[{source['index']}] Score: {source['score']:.3f}")
            print(f"Source: {source['source']}")
            print(f"Content: {source['content'][:150]}...")
        
        print("\n" + "="*60)
        print("✓ Query Test Successful!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Query Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("\n" + "="*60)
    print("Advanced RAG System - Test Suite")
    print("="*60)
    
    # Test components
    if not test_components():
        print("\n❌ Component tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. Run: python scripts/download_data.py --num-papers 50")
        print("2. Run: python scripts/build_index.py")
        print("3. Install Ollama: https://ollama.ai")
        print("4. Pull model: ollama pull llama3.1:8b")
        sys.exit(1)
    
    # Test query
    print("\n")
    input("Press Enter to test a sample query (this will take ~30 seconds)...")
    
    if not test_query():
        print("\n❌ Query test failed.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ All Tests Passed!")
    print("="*60)
    print("\nYour RAG system is ready to use!")
    print("\nNext steps:")
    print("1. Start the API server:")
    print("   python -m uvicorn src.api.main:app --reload")
    print("\n2. Open the API docs:")
    print("   http://localhost:8000/docs")
    print("\n3. Try more queries through the API!")


if __name__ == "__main__":
    main()
