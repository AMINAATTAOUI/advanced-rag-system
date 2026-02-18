"""
Smoke tests for the LangChain refactoring.

Validates:
  1. All LangChain imports resolve correctly
  2. Core classes can be instantiated (where no external service is needed)
  3. LangChain Document / ChatPromptTemplate / StrOutputParser work end-to-end
  4. Backward-compatible interfaces still exist

Run:  python -m pytest tests/test_langchain_imports.py -v
  or: python tests/test_langchain_imports.py
"""

import sys, os
from pathlib import Path

# ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────────────
# 1. Import checks — every LangChain package used in the codebase
# ─────────────────────────────────────────────────────────────────────────

def test_langchain_core_imports():
    """Core LangChain packages must be importable."""
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    print("  [OK] langchain_core imports")


def test_langchain_community_imports():
    """Community integrations used in the project."""
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    print("  [OK] langchain_community imports")


def test_langchain_integration_imports():
    """Dedicated LangChain integration packages."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama
    print("  [OK] langchain integration imports (text-splitters, chroma, huggingface, ollama)")


def test_langchain_retriever_imports():
    """Retriever / compressor abstractions."""
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    print("  [OK] langchain retriever / compressor imports")


def test_langchain_hf_llm_imports():
    """HuggingFace LLM imports for Inference API."""
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    print("  [OK] langchain HuggingFace LLM imports (ChatHuggingFace, HuggingFaceEndpoint)")


# ─────────────────────────────────────────────────────────────────────────
# 2. Source-module import checks — every refactored module loads
# ─────────────────────────────────────────────────────────────────────────

def test_src_data_imports():
    from src.data.loader import DocumentLoader
    from src.data.processor import DocumentProcessor, Chunk
    print("  [OK] src.data (loader, processor)")


def test_src_retrieval_imports():
    from src.retrieval.embeddings import EmbeddingModel
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import Reranker
    print("  [OK] src.retrieval (embeddings, vector_store, bm25, hybrid, reranker)")


def test_src_generation_imports():
    from src.generation.llm import LLMClient
    from src.generation.prompts import PromptTemplates
    from src.generation.chain import RAGChain
    print("  [OK] src.generation (llm, prompts, chain)")


def test_src_api_imports():
    from src.api.models import QueryRequest, QueryResponse, HealthResponse, StatsResponse
    print("  [OK] src.api.models")


# ─────────────────────────────────────────────────────────────────────────
# 3. Functional smoke tests (no external services required)
# ─────────────────────────────────────────────────────────────────────────

def test_document_creation():
    """LangChain Document creation works."""
    from langchain_core.documents import Document
    doc = Document(page_content="Hello world", metadata={"source": "test.txt"})
    assert doc.page_content == "Hello world"
    assert doc.metadata["source"] == "test.txt"
    print("  [OK] Document creation")


def test_prompt_template_formatting():
    """ChatPromptTemplate renders correctly."""
    from src.generation.prompts import PromptTemplates
    result = PromptTemplates.format_qa_prompt(
        query="What is AI?",
        context=["AI is artificial intelligence.", "It mimics human cognition."],
    )
    assert "What is AI?" in result
    assert "artificial intelligence" in result
    print("  [OK] ChatPromptTemplate formatting")


def test_prompt_template_langchain_objects():
    """Prompt templates are actual ChatPromptTemplate instances."""
    from langchain_core.prompts import ChatPromptTemplate
    from src.generation.prompts import PromptTemplates

    assert isinstance(PromptTemplates.QA_PROMPT, ChatPromptTemplate)
    assert isinstance(PromptTemplates.QA_WITH_SOURCES_PROMPT, ChatPromptTemplate)
    assert isinstance(PromptTemplates.SUMMARIZATION_PROMPT, ChatPromptTemplate)
    assert isinstance(PromptTemplates.MULTI_QUERY_PROMPT, ChatPromptTemplate)
    print("  [OK] All prompts are ChatPromptTemplate instances")


def test_str_output_parser():
    """StrOutputParser works standalone."""
    from langchain_core.output_parsers import StrOutputParser
    parser = StrOutputParser()
    from langchain_core.messages import AIMessage
    result = parser.invoke(AIMessage(content="test output"))
    assert result == "test output"
    print("  [OK] StrOutputParser")


def test_text_splitter():
    """RecursiveCharacterTextSplitter.from_tiktoken_encoder works."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=50,
        chunk_overlap=10,
    )
    chunks = splitter.split_text("Hello world. " * 100)
    assert len(chunks) > 1
    print(f"  [OK] Text splitter created {len(chunks)} chunks")


def test_document_processor():
    """DocumentProcessor (LangChain text splitter under the hood) processes docs."""
    from src.data.processor import DocumentProcessor

    proc = DocumentProcessor(chunk_size=50, chunk_overlap=10)
    doc = {"content": "Test sentence. " * 200, "metadata": {"source": "test.txt"}}
    chunks = proc.process_document(doc)
    assert len(chunks) > 0
    assert hasattr(chunks[0], "content")
    assert hasattr(chunks[0], "chunk_id")
    print(f"  [OK] DocumentProcessor produced {len(chunks)} Chunk objects")


def test_document_loader():
    """DocumentLoader can be instantiated and has LangChain loader method."""
    from src.data.loader import DocumentLoader

    loader = DocumentLoader()
    assert hasattr(loader, "_get_loader")
    assert hasattr(loader, "load_as_langchain_documents")
    print("  [OK] DocumentLoader has LangChain methods")


def test_embedding_model():
    """EmbeddingModel loads HuggingFaceEmbeddings and produces vectors."""
    from src.retrieval.embeddings import EmbeddingModel

    model = EmbeddingModel()
    assert hasattr(model, "lc_embeddings")

    # encode a short text
    embeddings = model.encode(["Hello world"])
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == 384  # bge-small-en-v1.5 dim
    print(f"  [OK] EmbeddingModel -> shape {embeddings.shape}")


def test_bm25_retriever_build_and_search():
    """BM25Retriever builds index and searches using LangChain BM25."""
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.data.processor import Chunk

    bm25 = BM25Retriever()
    chunks = [
        Chunk(content="Machine learning is great", metadata={"source": "a.txt"},
              chunk_id="c1", start_index=0, end_index=25, token_count=5),
        Chunk(content="Deep learning uses neural networks", metadata={"source": "b.txt"},
              chunk_id="c2", start_index=0, end_index=35, token_count=6),
    ]
    bm25.build_index(chunks)
    assert bm25.lc_retriever is not None

    results = bm25.search("machine learning", top_k=2)
    assert len(results) > 0
    assert "content" in results[0]
    print(f"  [OK] BM25Retriever search returned {len(results)} results")


def test_bm25_as_retriever():
    """BM25Retriever exposes a LangChain Retriever via as_retriever()."""
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.data.processor import Chunk
    from langchain_core.documents import Document

    bm25 = BM25Retriever()
    bm25.build_index([
        Chunk(content="Test doc one", metadata={}, chunk_id="t1",
              start_index=0, end_index=12, token_count=3),
        Chunk(content="Test doc two", metadata={}, chunk_id="t2",
              start_index=0, end_index=12, token_count=3),
    ])
    lc_ret = bm25.as_retriever()
    docs = lc_ret.invoke("test")
    assert all(isinstance(d, Document) for d in docs)
    print(f"  [OK] BM25 as_retriever().invoke() returned {len(docs)} Documents")


def test_prompt_get_system_prompt():
    """System prompt selector still works."""
    from src.generation.prompts import PromptTemplates
    assert "helpful" in PromptTemplates.get_system_prompt("default").lower()
    assert "research" in PromptTemplates.get_system_prompt("research").lower()
    assert "concise" in PromptTemplates.get_system_prompt("concise").lower()
    print("  [OK] get_system_prompt()")


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    # Import checks
    test_langchain_core_imports,
    test_langchain_community_imports,
    test_langchain_integration_imports,
    test_langchain_retriever_imports,
    test_langchain_hf_llm_imports,
    # Source module imports
    test_src_data_imports,
    test_src_retrieval_imports,
    test_src_generation_imports,
    test_src_api_imports,
    # Functional
    test_document_creation,
    test_prompt_template_formatting,
    test_prompt_template_langchain_objects,
    test_str_output_parser,
    test_text_splitter,
    test_document_processor,
    test_document_loader,
    test_embedding_model,
    test_bm25_retriever_build_and_search,
    test_bm25_as_retriever,
    test_prompt_get_system_prompt,
]


if __name__ == "__main__":
    print("=" * 65)
    print("  LangChain Refactoring Smoke Tests")
    print("=" * 65)

    passed, failed = 0, 0
    failures = []

    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            failures.append((test_fn.__name__, str(e)))
            print(f"  [FAIL] {test_fn.__name__}: {e}")

    print("\n" + "-" * 65)
    print(f"  Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    print("-" * 65)

    if failures:
        print("\nFailed tests:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("\n  All tests passed!")
        sys.exit(0)
