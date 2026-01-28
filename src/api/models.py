"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class QueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., description="User query", min_length=1)
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    use_reranking: Optional[bool] = Field(True, description="Whether to use reranking")
    include_sources: Optional[bool] = Field(True, description="Whether to include source information")
    temperature: Optional[float] = Field(0.1, description="LLM temperature", ge=0.0, le=2.0)
    stream: Optional[bool] = Field(False, description="Whether to stream the response")


class Source(BaseModel):
    """Source information for a retrieved document."""
    index: int
    content: str
    source: str
    score: float
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    answer: str
    sources: List[Source]
    query: str
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: Dict[str, bool]


class StatsResponse(BaseModel):
    """System statistics response."""
    vector_store: Dict[str, Any]
    bm25_index: Dict[str, Any]
    llm_model: str
    embedding_model: str


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
