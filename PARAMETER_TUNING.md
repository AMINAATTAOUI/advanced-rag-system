# Parameter Tuning Guide

This guide covers systematic parameter tuning and validation for your RAG system.

## Tuning Workflow

```
Baseline Capture → Parameter Testing → Validation → Production
```

## Phase 1: Baseline Capture

### Capture Initial Metrics

```bash
# Test with 10 PDFs first, or start with you min doc numbers based on you dwhole data size
python tests/test_baseline_metrics.py --dataset_size 10 --save_results

# Scale up to 50 PDFs
python scripts/download_data.py --num-papers 50
python scripts/build_index.py
python tests/test_baseline_metrics.py --dataset_size 50 --save_results

# Scale to 100 PDFs
python scripts/download_data.py --num-papers 100
python scripts/build_index.py
python tests/test_baseline_metrics.py --dataset_size 100 --save_results
```

### Check Scalability

```bash
# Compare 10 vs 50 PDFs
python tests/compare_scalability_baselines.py --compare
```

**Decision Gate:**
- OK Metrics stable (±5%) → Proceed to tuning
- NO Major degradation → Investigate issues first

##  Phase 2: Parameter Tuning

### Retrieval Parameters

#### Chunk Size & Overlap
```bash
# Test different chunk configurations
# Edit .env file:
CHUNK_SIZE=256    # Try: 256, 512, 1024
CHUNK_OVERLAP=25  # Try: 25, 50, 75

# Rebuild and test
python scripts/build_index.py --reset
python tests/test_baseline_metrics.py --dataset_size 50
```

#### Retrieval Weights
```bash
# Edit .env file:
DENSE_WEIGHT=0.7  # Semantic search weight (0.0-1.0)
SPARSE_WEIGHT=0.3  # Keyword search weight (0.0-1.0)

# Test combinations:
# Dense-focused: 0.8/0.2
# Balanced: 0.6/0.4
# Sparse-focused: 0.4/0.6
```

#### Top-K Values
```bash
# Edit .env file:
TOP_K_RETRIEVAL=15  # Initial retrieval (try: 10, 15, 20)
TOP_K_RERANK=5      # After reranking (try: 3, 5, 7)
```

### Generation Parameters

#### LLM Model Selection
```bash
# Try different models (edit .env):
OLLAMA_MODEL=llama3.1:3b   # Fast, less accurate
OLLAMA_MODEL=llama3.1:8b   # Balanced (default)
OLLAMA_MODEL=llama3.1:70b  # Slow, most accurate
```

#### Temperature & Sampling
```bash
# Edit .env:
LLM_TEMPERATURE=0.1  # Try: 0.0, 0.1, 0.3, 0.7
LLM_TOP_P=0.9        # Try: 0.9, 0.95, 1.0
```

### Automated Tuning Script

```bash
# Run systematic parameter search
python scripts/tune_parameters.py --param chunk_size --values 256,512,1024
python scripts/tune_parameters.py --param dense_weight --values 0.4,0.6,0.8
python scripts/tune_parameters.py --param top_k --values 3,5,7,10
```

## Phase 3: Validation Testing

### Comprehensive Evaluation

```bash
# Run all evaluation metrics
python tests/test_evaluate_retrieval.py --method hybrid --top_k 5
python tests/test_evaluate_generation.py --method hybrid --num_queries 20
```

### Failure Analysis

```bash
# Analyze systematic failures
python scripts/failure_analysis.py --threshold 0.6
```

**Check results/failure_analysis.json for insights**

### Score Threshold Tuning

```bash
# Test different score thresholds
python tests/test_score_thresholds.py --method hybrid --thresholds 0.1,0.3,0.5,0.7
```

## Phase 4: Production Optimization

### Performance Tuning

```bash
# Enable caching
REDIS_ENABLED=true
REDIS_URL=redis://localhost:6379

# GPU acceleration (if available)
EMBEDDING_DEVICE=cuda
OLLAMA_GPU_LAYERS=35
```

### Final Validation

```bash
# Production readiness test
python tests/test_system.py --production_check

# Load testing
python tests/test_system.py --load_test --num_queries 100
```

## Key Metrics to Monitor

### Retrieval Metrics, targets based on used dataset for test
- **Precision@5**: > 0.85 (target)
- **Recall@5**: > 0.75 (target)
- **MRR**: > 0.85 (target)
- **Latency**: < 2s (target)

### Generation Metrics
- **Faithfulness**: > 0.9 (target)
- **Answer Relevance**: > 0.85 (target)
- **Hallucination Rate**: < 0.1 (target)

### System Metrics
- **Total Response Time**: < 8s (target)
- **Memory Usage**: < 8GB (target)
- **Error Rate**: < 1% (target)

##  Troubleshooting Tuning Issues

### Poor Retrieval Performance
```bash
# Check chunk quality
python scripts/failure_analysis.py --focus retrieval

# Possible fixes:
# - Increase chunk overlap
# - Use domain-specific embeddings
# - Adjust BM25 parameters
```

### High Latency
```bash
# Profile bottlenecks
python tests/test_system.py --profile

# Possible fixes:
# - Reduce top_k values
# - Enable caching
# - Use smaller LLM model
```

### Generation Issues
```bash
# Check hallucination sources
python scripts/failure_analysis.py --focus generation

# Possible fixes:
# - Improve context quality
# - Adjust temperature
# - Use stricter prompts
```

## Tuning Checklist

- [ ] Baseline captured (10, 50, 100 PDFs)
- [ ] Scalability verified (±5% metrics)
- [ ] Chunk size optimized (256-1024)
- [ ] Retrieval weights balanced (dense/sparse)
- [ ] Top-K values tuned (3-10)
- [ ] LLM model selected (3b/8b/70b)
- [ ] Temperature optimized (0.0-0.7)
- [ ] Caching enabled
- [ ] Production validation passed
- [ ] Performance targets met

##  Next Steps

After tuning:
1. **Deploy to production** (see GETTING_STARTED.md)
2. **Monitor performance** in production
3. **A/B test** different configurations
4. **Iterate** based on user feedback

---

**Remember**: Tuning is iterative. Start with baselines, make one change at a time, measure impact, then iterate.
