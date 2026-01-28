"""
FastAPI application for the RAG system.
Provides REST API endpoints for querying the system.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
from typing import AsyncIterator

from src.api.models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse
)
from src.generation.chain import RAGChain
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import Reranker
from src.generation.llm import LLMClient
from src.config import settings
from src.utils.logger import log
from src import __version__


# Global instances
rag_chain = None
vector_store = None
bm25_retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    log.info("Starting RAG API server...")
    
    global rag_chain, vector_store, bm25_retriever
    
    try:
        # Initialize components
        log.info("Initializing components...")
        vector_store = VectorStore()
        bm25_retriever = BM25Retriever()
        
        # Load BM25 index if exists
        if not bm25_retriever.load_index():
            log.warning("BM25 index not found. Please run build_index.py first.")
        
        # Initialize reranker
        reranker = Reranker()
        
        # Initialize hybrid retriever
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            reranker=reranker
        )
        
        # Initialize LLM client
        llm_client = LLMClient()
        
        # Initialize RAG chain
        rag_chain = RAGChain(
            retriever=hybrid_retriever,
            llm_client=llm_client
        )
        
        log.info("RAG API server started successfully!")
        
    except Exception as e:
        log.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    log.info("Shutting down RAG API server...")


# Create FastAPI app
app = FastAPI(
    title="Advanced RAG System API",
    description="Production-ready RAG system with hybrid retrieval and reranking",
    version=__version__,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Advanced RAG System API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check components
        components = {
            "vector_store": vector_store is not None and vector_store.collection.count() > 0,
            "bm25_index": bm25_retriever is not None and bm25_retriever.bm25 is not None,
            "llm": rag_chain is not None and rag_chain.llm_client.check_model_availability(),
            "rag_chain": rag_chain is not None
        }
        
        all_healthy = all(components.values())
        
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            version=__version__,
            components=components
        )
    except Exception as e:
        log.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    try:
        vector_stats = vector_store.get_stats()
        bm25_stats = bm25_retriever.get_stats()
        
        return StatsResponse(
            vector_store=vector_stats,
            bm25_index=bm25_stats,
            llm_model=settings.ollama_model,
            embedding_model=settings.embedding_model
        )
    except Exception as e:
        log.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    Args:
        request: Query request with parameters
        
    Returns:
        Query response with answer and sources
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    
    try:
        log.info(f"Received query: {request.query[:100]}...")
        
        # Execute query
        result = rag_chain.query(
            query=request.query,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
            include_sources=request.include_sources,
            temperature=request.temperature
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        log.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Query the RAG system with streaming response.
    
    Args:
        request: Query request with parameters
        
    Returns:
        Streaming response with answer chunks
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    
    async def generate() -> AsyncIterator[str]:
        try:
            log.info(f"Received streaming query: {request.query[:100]}...")
            
            for chunk in rag_chain.query_stream(
                query=request.query,
                top_k=request.top_k,
                use_reranking=request.use_reranking,
                temperature=request.temperature
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
                
        except Exception as e:
            log.error(f"Error in streaming query: {e}")
            error_chunk = {
                "type": "error",
                "content": str(e),
                "done": True
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    log.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
