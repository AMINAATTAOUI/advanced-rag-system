"""
Configuration management for the RAG system.
Loads settings from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Configuration
    llm_backend: str = Field(default="huggingface", env="LLM_BACKEND")  # "huggingface" or "ollama"
    
    # HuggingFace Inference API
    hf_model: str = Field(default="meta-llama/Llama-3.1-8B-Instruct", env="HF_MODEL")
    hf_api_token: str = Field(default="", env="HF_API_TOKEN")
    
    # Ollama (fallback)
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
    
    # Embedding Configuration
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", env="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", env="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    
    # Vector Store Configuration
    vector_db_path: str = Field(default="./data/vector_db", env="VECTOR_DB_PATH")
    collection_name: str = Field(default="arxiv_papers", env="COLLECTION_NAME")
    
    # Reranker Configuration
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        env="RERANKER_MODEL"
    )
    
    # Retrieval Settings
    top_k_retrieval: int = Field(default=40, env="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(default=10, env="TOP_K_RERANK")
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    
    # Hybrid Search Weights
    dense_weight: float = Field(default=0.7, env="DENSE_WEIGHT")
    sparse_weight: float = Field(default=0.3, env="SPARSE_WEIGHT")
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/rag_system.log", env="LOG_FILE")
    
    # Cache Settings
    enable_cache: bool = Field(default=True, env="ENABLE_CACHE")
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # Dataset Configuration
    arxiv_categories: str = Field(
    default="cs.AI,cs.CL,cs.LG",
    env="ARXIV_CATEGORIES"
    )
    
    @field_validator('arxiv_categories', mode='after')
    @classmethod
    def parse_arxiv_categories(cls, v):
        """Parse arxiv_categories from string or list."""
        if isinstance(v, str):
            # Split by comma and strip whitespace
            return [cat.strip() for cat in v.split(',')]
        return v
    num_papers: int = Field(default=100, env="NUM_PAPERS")
    data_raw_path: str = Field(default="./data/raw", env="DATA_RAW_PATH")
    test_data_path: str = Field(default="./data/test", env="TEST_DATA_PATH")
    data_processed_path: str = Field(default="./data/processed", env="DATA_PROCESSED_PATH")
    
    # Generation Settings
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    max_tokens: int = Field(default=512, env="MAX_TOKENS")
    
    # Evaluation Settings
    eval_precision_k: int = Field(default=5, env="EVAL_PRECISION_K")
    eval_recall_k: int = Field(default=5, env="EVAL_RECALL_K")
    eval_adaptive: bool = Field(default=True, env="EVAL_ADAPTIVE")
    eval_small_dataset_threshold: int = Field(default=50, env="EVAL_SMALL_DATASET_THRESHOLD")
    eval_medium_dataset_threshold: int = Field(default=200, env="EVAL_MEDIUM_DATASET_THRESHOLD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        os.makedirs(self.data_raw_path, exist_ok=True)
        os.makedirs(self.vector_db_path, exist_ok=True)
        os.makedirs(self.test_data_path, exist_ok=True)
        os.makedirs(self.data_processed_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
