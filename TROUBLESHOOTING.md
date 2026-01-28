# Troubleshooting Guide

Common issues and solutions for the Advanced RAG System (works with any dataset).

## Quick Diagnosis

### System Health Check

```bash
# Check if system is running
curl http://localhost:8000/health

# Run system test
python tests/test_system.py

# Check logs
tail -f logs/rag_system.log
```

### Common Symptoms & Solutions

| Symptom | Likely Cause | Quick Fix |
|---------|-------------|-----------|
| "Ollama not found" | Ollama not installed/running | Install/start Ollama |
| "No documents found" | Index not built | Run `python scripts/build_index.py` |
| Slow responses | Large top_k or big model | Reduce `TOP_K_RETRIEVAL` or use smaller model |
| Out of memory | Too many documents | Reduce dataset size or increase RAM | Or use more GPUs
| Poor answers | Wrong parameters or low-quality source data | Check `PARAMETER_TUNING.md` or review your dataset |

##  Setup Issues

### Ollama Problems

**Issue:** `ollama: command not found`
```bash
# Solution: Install Ollama
# Windows: Download from https://ollama.ai/download
# Then verify:
ollama --version
```

**Issue:** `model not found`
```bash
# Pull the model
ollama pull llama3.1:8b

# List available models
ollama list
```

**Issue:** `connection refused`
```bash
# Start Ollama server
ollama serve

# Or check if it's running
curl http://localhost:11434/api/tags
```

### Dependency Issues

**Issue:** `ModuleNotFoundError`
```bash
# Install missing dependencies
pip install -r requirements.txt

# Upgrade pip if needed
python -m pip install --upgrade pip
```

**Issue:** `ImportError: No module named 'chromadb'`
```bash
# Force reinstall
pip uninstall chromadb
pip install chromadb
```

### Environment Issues

**Issue:** `.env file not found`
```bash
# Copy example file
copy .env.example .env

# Or create manually
echo "OLLAMA_MODEL=llama3.1:8b" > .env
echo "CHUNK_SIZE=512" >> .env
```

## Data & Indexing Issues

### Data Loading Problems

**Issue:** `No documents found`
```bash
# For example dataset (research papers):
python scripts/download_data.py --num-papers 10

# For your own data:
# Ensure files are in data/raw/ directory
# Then rebuild the index
dir data\raw\
python scripts/build_index.py
```

**Issue:** `ArXiv API rate limited`
```bash
# Wait a few minutes, then retry
# Or reduce batch size in downloader.py
# (only applies to example dataset)
```

### Indexing Failures

**Issue:** `Index build failed`
```bash
# Check available disk space
# Windows: wmic logicaldisk get size,freespace,caption

# Clear old indices
rmdir /s data\vector_db
rmdir /s data\processed

# Retry with your data in data/raw/
python scripts/build_index.py
```
python scripts/build_index.py
```

**Issue:** `Out of memory during indexing`
```bash
# Reduce batch size in .env
EMBEDDING_BATCH_SIZE=8  # Try 4 or 2

# Or use smaller dataset
python scripts/download_data.py --num-papers 25
```

**Issue:** `BM25 index corrupted`
```bash
# Delete and rebuild
del data\processed\bm25_index.pkl
python scripts/build_index.py
```

## Retrieval Issues

### Poor Retrieval Quality

**Issue:** Irrelevant documents retrieved
```bash
# Check parameters
# Edit .env:
TOP_K_RETRIEVAL=20  # Increase initial retrieval
TOP_K_RERANK=5      # Keep reranking tight

# Rebuild index if needed
python scripts/build_index.py --reset
```

**Issue:** Missing relevant documents
```bash
# Analyze failures
python scripts/failure_analysis.py --focus retrieval

# Possible fixes:
# - Increase chunk overlap
# - Adjust dense/sparse weights
# - Use domain-specific embeddings
```

### Performance Issues

**Issue:** Retrieval too slow (>2s)
```bash
# Profile bottlenecks
python tests/test_system.py --profile

# Fixes:
# - Reduce TOP_K_RETRIEVAL
# - Enable caching: REDIS_ENABLED=true
# - Use GPU: EMBEDDING_DEVICE=cuda
```

# Generation Issues

### LLM Response Problems

**Issue:** Empty or nonsense answers
```bash
# Check LLM is responding
curl http://localhost:11434/api/generate -d '{"model":"llama3.1:8b","prompt":"test"}'

# Verify model loaded
ollama list
```

**Issue:** Hallucinations (wrong information)
```bash
# Analyze generation failures
python scripts/failure_analysis.py --focus generation

# Fixes:
# - Improve context quality (better retrieval)
# - Reduce temperature: LLM_TEMPERATURE=0.1
# - Use stricter prompts
```

**Issue:** Responses too slow (>10s)
```bash
# Try smaller model
# Edit .env:
OLLAMA_MODEL=llama3.1:3b

# Or reduce context length
MAX_CONTEXT_LENGTH=2048
```

### Streaming Issues

**Issue:** Streaming not working
```bash
# Check API endpoint
curl -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "stream": true}'

# Verify client supports SSE
```

## API Issues

### Connection Problems

**Issue:** `Connection refused`
```bash
# Check if server is running
netstat -ano | findstr :8000

# Start server
python -m uvicorn src.api.main:app --reload --port 8000
```

**Issue:** `CORS errors`
```bash
# Add CORS middleware in src/api/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Request Issues

**Issue:** `422 Validation Error`
```json
// Check request format
{
  "query": "Your question here",
  "top_k": 5,
  "use_reranking": true
}
```

**Issue:** `429 Too Many Requests`
```bash
# Wait before retrying
# Or increase rate limits in .env
RATE_LIMIT_REQUESTS=20
RATE_LIMIT_WINDOW=60
```

## Storage Issues

### Disk Space Problems

**Issue:** `No space left on device`
```bash
# Check disk usage
# Windows: wmic logicaldisk get size,freespace,caption

# Clean up:
# Remove old logs
del logs\*.log

# Clear cache
# POST /admin/clear-cache

# Reduce dataset size
python scripts/download_data.py --num-papers 50
```

### Database Corruption

**Issue:** `ChromaDB corrupted`
```bash
# Backup and reset
xcopy data\vector_db data\vector_db_backup /E /I /H /Y
rmdir /s data\vector_db
python scripts/build_index.py
```

## Performance Tuning

### Memory Optimization

```bash
# Edit .env for lower memory usage
EMBEDDING_BATCH_SIZE=4
MAX_WORKERS=2
CHUNK_SIZE=256
OLLAMA_NUM_GPU_LAYERS=0  # CPU only
```

### Speed Optimization

```bash
# Edit .env for faster responses
TOP_K_RETRIEVAL=10
OLLAMA_MODEL=llama3.1:3b
REDIS_ENABLED=true
CACHE_TTL_SECONDS=3600
```

## Monitoring & Debugging

### Log Analysis

```bash
# View recent logs
tail -f logs/rag_system.log

# Search for errors
findstr ERROR logs\rag_system.log

# Check specific component
grep "retrieval" logs/rag_system.log
```

### Performance Monitoring

```bash
# Enable metrics
# GET /metrics

# Profile specific query
python tests/test_system.py --profile --query "test query"
```

### Health Checks

```bash
# Comprehensive health check
python tests/test_system.py --full_check

# Individual component tests
python tests/test_embeddings.py
python tests/test_retrieval.py
python tests/test_generation.py
```

## Critical Issues

### System Won't Start

**Symptoms:** Server crashes immediately
```bash
# Check Python version
python --version  # Should be 3.9+

# Check dependencies
pip check

# Run with debug
python -c "import src.config; print('Config OK')"

# Basic system test
python tests/test_system.py
```

### Data Corruption

**Symptoms:** Inconsistent results, crashes
```bash
# Full reset
rmdir /s data
python scripts/download_data.py --num-papers 10
python scripts/build_index.py
```

### Complete Failure Recovery

```bash
# Nuclear option - fresh start
# Backup important files first
mkdir backup
copy .env backup\
copy results\*.json backup\

# Clean everything
rmdir /s data
rmdir /s logs
del .env

# Restart from scratch
copy .env.example .env
# Follow GETTING_STARTED.md
```

## Getting Help

### Debug Information to Provide

When asking for help, include:

```bash
# System info
python -c "import sys; print(f'Python: {sys.version}')"

# Environment
set | grep -E "(OLLAMA|CHUNK|TOP_K)"

# Logs
tail -20 logs/rag_system.log

# Test results
python test_system.py
```

### Common Debug Commands

```bash
# Check all services
ollama list
curl http://localhost:8000/health
dir data\vector_db
dir data\processed

# Performance test
python tests/test_system.py --benchmark

# Configuration validation
python -c "from src.config import settings; print(settings.dict())"
```

## Support Resources

- **API Documentation**: `http://localhost:8000/docs`
- **Configuration Guide**: Check `.env.example`
- **Parameter Tuning**: See `PARAMETER_TUNING.md`
- **Setup Guide**: See `GETTING_STARTED.md`

---

**Most issues can be resolved by checking logs and running the system test. Start with `python tests/test_system.py` for automated diagnosis.**
