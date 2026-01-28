# Advanced RAG System

Production-ready foundation and tutorial for building Retrieval-Augmented Generation systems. Deploy with your own datasets (documents, resumes, knowledge bases, etc.). 

**Example**: We demonstrate the system using ArXiv research papers, but it works with any document collection.

## Features

- **Hybrid Search**: Dense (semantic) + sparse (BM25) retrieval for better relevance
- **Cross-Encoder Reranking**: Re-rank results to improve answer quality
- **Local LLM**: Llama 3.1 via Ollama (free, private, no API costs, chose your model instead)
- **Production API**: FastAPI with streaming and health checks
- **Complete Evaluation Suite**: Metrics, baselines, failure analysis, parameter tuning
- **Vector Database**: ChromaDB with BGE embeddings
- **Extensible Foundation**: Easily adapt to your own data sources and use cases

## Architecture

```
Query → Hybrid Retrieval (ChromaDB + BM25) → Reranking → Llama 3.1 → Response
```

## Tech Stack

- **LLM**: Llama 3.1 (Meta, via Ollama)
- **Embeddings**: BGE-small-en-v1.5 (Alibaba BAAI, from Hugging Face)
- **Vector DB**: ChromaDB (open-source)
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2 (Microsoft MiniLM via Hugging Face)
- **Framework**: LangChain (open-source)
- **API**: FastAPI (open-source)

## Quick Start (With Example Dataset)

Example using ArXiv research papers:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Ollama from https://ollama.ai/download
ollama pull llama3.1:8b

# 3. Download example dataset (research papers)
python scripts/download_data.py --num-papers 10
python scripts/build_index.py

# 4. Test the system
python tests/test_system.py

# 5. Start API server
python -m uvicorn src.api.main:app --reload --port 8000
```

Visit http://localhost:8000/docs for interactive API documentation.

**To use your own data**: See [GETTING_STARTED.md](GETTING_STARTED.md#-adding-your-own-data) for custom datasets section.

## Documentation

| Guide | Purpose |
|-------|---------|
| **[GETTING_STARTED.md](GETTING_STARTED.md)** | Complete setup → testing → tuning → production workflow |
| **[PARAMETER_TUNING.md](PARAMETER_TUNING.md)** | Systematic parameter tuning, evaluation, validation |
| **[API_REFERENCE.md](API_REFERENCE.md)** | REST API endpoints and examples |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Common issues and solutions |

## Project Structure

```
advanced-rag-system/
├── src/
│   ├── data/          # Document loading and processing
│   ├── retrieval/     # Hybrid search, embeddings, reranking
│   ├── generation/    # LLM interface and prompts
│   ├── api/           # FastAPI endpoints
│   └── utils/         # Logging, caching
├── scripts/
│   ├── download_data.py       # Download ArXiv papers
│   ├── build_index.py         # Build vector/BM25 indices
│   ├── tune_parameters.py     # Automated parameter tuning
│   └── failure_analysis.py    # Analyze system failures
├── tests/
│   ├── test_system.py                 # End-to-end system test
│   ├── test_baseline_metrics.py       # Capture baseline metrics
│   ├── test_evaluate_retrieval.py     # Retrieval evaluation
│   ├── test_evaluate_generation.py    # Generation evaluation
│   ├── test_score_thresholds.py       # Threshold tuning
│   └── compare_scalability_baselines.py  # Scalability analysis
├── data/
│   ├── raw/           # Source documents
│   ├── processed/     # Chunked documents
│   └── vector_db/     # ChromaDB storage
└── results/           # Evaluation metrics and analyses
```

## Key Commands

### Data & Indexing
```bash
# Download example dataset (research papers)
python scripts/download_data.py --num-papers 50

# Build indices from your data
python scripts/build_index.py --reset
```

### Query Examples
```bash
# Example with research papers
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are transformers?", "top_k": 5}'

# Example with resumes/documents
# curl ... -d '{"query": "List experience with machine learning", "top_k": 5}'
```

### Testing & Evaluation
```bash
python tests/test_system.py
# The data_size 50 is the number of raw files, it help to adaptative top-k metrics
python tests/test_baseline_metrics.py --dataset_size 50 --save_results
# evaluation of retrieval and generation based on 20 queries dataset designed to performance test and evaluation
python tests/test_evaluate_retrieval.py --method hybrid,dense,sparse --top_k 5
python tests/test_evaluate_generation.py --method hybrid,dense,sparse  --num_queries 20
```

### Parameter Tuning & Optimization
```bash
python scripts/tune_parameters.py --param chunk_size --values 256,512,1024
python scripts/failure_analysis.py --threshold 0.6
python tests/test_score_thresholds.py --thresholds 0.3,0.5,0.7
```

## Target Performance

- Retrieval Precision@5: > 0.85
- Answer Faithfulness: > 0.9
- Response Time: < 8s
- Memory Usage: < 8GB

## License

MIT License
