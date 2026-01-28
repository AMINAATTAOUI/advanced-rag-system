# Complete Getting Started Guide

This guide covers the complete journey from setup to production. Example uses ArXiv research papers, but you can use your own data (documents, resumes, knowledge bases, etc.).

## Prerequisites Check

**Default Configuration (CPU - Recommended for most users)**
-  **Python 3.9+** installed
-  **RAM**: 8GB+ (minimum 4GB for small datasets)
-  **Disk space**: 15GB+ (Ollama model 4.7GB + indices + OS)
-  **Windows/Linux/macOS**

**For GPU Acceleration (Optional)**
- Add `EMBEDDING_DEVICE=cuda` to `.env` (requires NVIDIA GPU + CUDA)
- Reduces memory requirements and improves embedding speed
- RAM: 4GB+ (VRAM handles computation)

##  Phase 1: Initial Setup (20 minutes)

### 1. Environment Setup

```bash
# Navigate to project
cd advanced-rag-system

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama (Local LLM) or you prefered LLMs provider

```bash
# Download from https://ollama.ai/download
# Install and run Ollama

# Verify installation
ollama --version

# Pull the model (4.7GB download)
ollama pull llama3.1:8b

# Verify model available
ollama list
```

### 4. Create Configuration

```bash
# Copy environment template
copy .env.example .env
```

##  Phase 2: Data & Indexing (30 minutes)

### Download Example Dataset (or use your own)

**Option A: Example Dataset (ArXiv research papers)**

```bash
# Start with 10 papers for quick testing
python scripts/download_data.py --num-papers 10

# Verify download
dir data\raw
```

**Option B: Your Own Data**

```bash
# Place your documents in data/raw/
# Supported: PDF, TXT, markdown, etc.
copy "your_documents/*" data\raw\
```

### Build Search Indices

```bash
# Build vector and BM25 indices from data/raw/
python scripts/build_index.py

# This creates:
# - data/processed/ (chunked documents)
# - data/vector_db/ (ChromaDB vectors)
# - BM25 index for keyword search
```

**Expected time:** 5-10 minutes for 10 documents

##  Phase 3: System Testing (10 minutes)

### Start API Server

```bash
# Start FastAPI server
python -m uvicorn src.api.main:app --reload --port 8000
```

**Server URLs:**
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### Health Check

```bash
# Check system status
curl http://localhost:8000/health
```

### First Query Test

```bash
# Generic test query (adapt to your dataset)
curl -X POST "http://localhost:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"Your question here\"}"

# Example: Research papers dataset
# curl ... -d "{\"query\": \"What are transformers in machine learning?\"}"

# Example: Resumes dataset
# curl ... -d "{\"query\": \"List experience with machine learning\"}"

# Example: Documentation dataset
# curl ... -d "{\"query\": \"How to configure authentication?\"}"
```

**Expected response:**
```json
{
  "answer": "...",
  "sources": [...],
  "query": "Your question here"
}
```

### Automated System Test

```bash
# Run comprehensive test
python test_system.py
```

## Phase 4: Baseline Evaluation (15 minutes)

### Capture Initial Metrics

```bash
# Test with current dataset (10 documents)
python tests/test_baseline_metrics.py --dataset_size 10 --save_results
```

**This creates:** `results/baseline_metrics_10docs.json`

### Review Results

```bash
# Check the results file
type results\baseline_metrics_10docs.json
```

**Key metrics to verify:**
- Retrieval Precision@3: > 0.8
- Generation Faithfulness: > 0.8
- System health: READY_FOR_TUNING

## Phase 5: Parameter Tuning (45 minutes)

### Scale Dataset

```bash
# Add more documents for better testing
# For example dataset (research papers)
python scripts/download_data.py --num-papers 50
python scripts/build_index.py

# For your own data: add more files to data/raw/ then rebuild
python scripts/build_index.py --reset
```

### Test Scalability

```bash
# Capture 50-document baseline
python tests/test_baseline_metrics.py --dataset_size 50 --save_results

# Compare scalability (10 vs 50 documents)
python tests/compare_scalability_baselines.py --compare
```

**Decision Gate:**
- OK Metrics stable (±5%) → Continue tuning
- NO Major degradation → Investigate issues

### Systematic Tuning

```bash
# Test different chunk sizes
python scripts/tune_parameters.py --param chunk_size --values 256,512,1024

# Test retrieval weights
python scripts/tune_parameters.py --param dense_weight --values 0.6,0.7,0.8

# Test top-k values
python scripts/tune_parameters.py --param top_k --values 3,5,7
```

### LLM Model Comparison

```bash
# Try different models (edit .env)
OLLAMA_MODEL=llama3.1:3b   # Faster, less accurate
OLLAMA_MODEL=llama3.1:8b   # Balanced (default)
OLLAMA_MODEL=llama3.1:70b  # Slower, most accurate

# Test each model
python tests/test_baseline_metrics.py --dataset_size 50
```

### Failure Analysis

```bash
# Analyze systematic issues
python scripts/failure_analysis.py --threshold 0.6

# Review results/failure_analysis.json
```

##  Phase 6: Validation & Optimization (30 minutes)

### Comprehensive Evaluation

```bash
# Run all evaluation metrics
python tests/test_evaluate_retrieval.py --method hybrid --top_k 5
python tests/test_evaluate_generation.py --method hybrid --num_queries 20
```

### Score Threshold Tuning

```bash
# Test different thresholds
python tests/test_score_thresholds.py --method hybrid --thresholds 0.1,0.3,0.5
```

### Performance Optimization

```bash
# Enable caching for production
# Edit .env:
REDIS_ENABLED=true
CACHE_TTL_SECONDS=3600

# GPU acceleration (if available)
EMBEDDING_DEVICE=cuda
OLLAMA_GPU_LAYERS=35
```

## Phase 7: Production Deployment (20 minutes)

### Final Validation

```bash
# Production readiness test
python tests/test_system.py --production_check

# Load testing
python tests/test_system.py --load_test --num_queries 100
```

### Scale to Production Dataset

```bash
# For example dataset (research papers)
python scripts/download_data.py --num-papers 100
python scripts/build_index.py

# For your own data: ensure all documents are in data/raw/
# Then rebuild the index
python scripts/build_index.py --reset

# Final baseline metrics
python tests/test_baseline_metrics.py --dataset_size 100 --save_results
```
# Download full dataset
python scripts/download_data.py --num-papers 100
python scripts/build_index.py

# Final baseline
python tests/test_baseline_metrics.py --dataset_size 100 --save_results
```

### Production Configuration

```bash
# Edit .env for production
RATE_LIMIT_REQUESTS=50
RATE_LIMIT_WINDOW=60
LOG_LEVEL=INFO
MAX_WORKERS=4

# Start production server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

##  Monitoring & Maintenance

### Health Monitoring

```bash
# Regular health checks
curl http://localhost:8000/health

# Performance monitoring
curl http://localhost:8000/metrics
```

### Log Analysis

```bash
# Monitor logs
tail -f logs/rag_system.log

# Search for errors
findstr ERROR logs\rag_system.log
```

### Backup Strategy

```bash
# Backup important data
xcopy results results_backup /E /I /H /Y
xcopy data\vector_db data_backup /E /I /H /Y

# Backup configuration
copy .env .env.backup
```

##  Adding Your Own Data

### Custom Documents

```bash
# Place your PDFs in data/raw/
copy "your_document.pdf" data\raw\

# Rebuild indices
python scripts/build_index.py --reset

# Test with your data
curl -X POST "http://localhost:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"your question about the document\"}"
```

### Custom Data Sources

```bash
# Modify scripts/download_data.py
# Add your own document sources
# Update data loading in src/data/loader.py
```

##  Complete Testing & Validation Reference

All test scripts and validation tools available in the system:

### Core System Tests

```bash
# End-to-end system validation
python tests/test_system.py

# Test embeddings
python tests/test_embeddings.py

# Test chunking strategies
python tests/test_chunking_run.py
```

### Baseline & Metrics

```bash
# Capture baseline metrics (10 papers)
python tests/test_baseline_metrics.py --dataset_size 10 --save_results

# Capture baseline metrics (50 papers)
python tests/test_baseline_metrics.py --dataset_size 50 --save_results

# Capture baseline metrics (100 papers)
python tests/test_baseline_metrics.py --dataset_size 100 --save_results

# Compare scalability across baselines
python tests/compare_scalability_baselines.py
```

### Retrieval Evaluation

```bash
# Evaluate hybrid retrieval
python tests/test_evaluate_retrieval.py --method hybrid --top_k 5

# Evaluate dense (semantic) only
python tests/test_evaluate_retrieval.py --method dense --top_k 5

# Evaluate sparse (BM25) only
python tests/test_evaluate_retrieval.py --method sparse --top_k 5
```

### Generation Evaluation

```bash
# Evaluate generation quality
python tests/test_evaluate_generation.py --method hybrid --num_queries 20

# Evaluate with more queries for statistical significance
python tests/test_evaluate_generation.py --method hybrid --num_queries 50
```

### Threshold Tuning

```bash
# Test score thresholds
python tests/test_score_thresholds.py --method hybrid --thresholds 0.1,0.3,0.5,0.7

# Find optimal threshold for precision/recall balance
python tests/test_score_thresholds.py --method hybrid --thresholds 0.2,0.25,0.3,0.35,0.4
```

### Parameter Tuning Scripts

```bash
# Systematic parameter search - chunk size
python scripts/tune_parameters.py --param chunk_size --values 256,512,1024

# Systematic parameter search - dense/sparse weights
python scripts/tune_parameters.py --param dense_weight --values 0.4,0.6,0.7,0.8

# Systematic parameter search - top-k retrieval
python scripts/tune_parameters.py --param top_k --values 3,5,7,10
```

### Failure Analysis

```bash
# Analyze all failures with default threshold (0.6)
python scripts/failure_analysis.py

# Analyze failures with custom threshold
python scripts/failure_analysis.py --threshold 0.5

# Focus on retrieval failures
python scripts/failure_analysis.py --focus retrieval

# Focus on generation failures
python scripts/failure_analysis.py --focus generation
```

See `results/` folder for all generated metrics and analysis files.

# Advanced Usage Examples

### Streaming Responses

```python
import requests

# Enable streaming
response = requests.post(
    "http://localhost:8000/query/stream",
    json={"query": "Explain transformers", "stream": True},
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### Batch Processing

```python
import requests

queries = [
    "What is attention mechanism?",
    "How does BERT work?",
    "Compare GPT and BERT"
]

for query in queries:
    response = requests.post(
        "http://localhost:8000/query",
        json={"query": query, "top_k": 3}
    )
    print(f"Q: {query}")
    print(f"A: {response.json()['answer'][:100]}...")
    print()
```

### Custom Parameters

```python
# Advanced query with custom settings
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What are transformers?",
        "top_k": 10,
        "use_reranking": True,
        "temperature": 0.1,
        "max_tokens": 500
    }
)
```

## Success Checklist

- [ ] Environment setup complete
- [ ] Ollama installed and model pulled
- [ ] Data downloaded and indexed
- [ ] API server running
- [ ] Basic queries working
- [ ] Baseline metrics captured
- [ ] Parameter tuning completed
- [ ] Validation tests passed
- [ ] Production deployment ready
- [ ] Monitoring configured

##  Getting Help

### Quick Diagnosis

```bash
# Run system test for automated diagnosis
python tests/test_system.py

# Check logs for errors
tail -20 logs/rag_system.log
```

### Common Issues

- **"Ollama not found"** → `ollama serve`
- **"No documents"** → Run download & build scripts
- **"Slow responses"** → Reduce top_k or use smaller model
- **"Out of memory"** → Reduce batch size or dataset size

### Support Resources

- **API Documentation**: http://localhost:8000/docs
- **Parameter Tuning Guide**: See `PARAMETER_TUNING.md`
- **Troubleshooting**: See `TROUBLESHOOTING.md`
- **Logs**: `logs/rag_system.log`


**Happy building! **
