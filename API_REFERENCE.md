# API Reference

REST API for the Advanced RAG System. Query any dataset you've indexed.

## Server Startup

Development:

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

Production (example):

```bash
pip install gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Base URL: http://localhost:8000

Interactive docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## POST /query

Query your indexed dataset (research papers, resumes, documents, knowledge bases, etc.).

Request (JSON):

```json
{
  "query": "Your question here (required)",
  "top_k": 5,
  "use_reranking": true,
  "temperature": 0.1,
  "max_tokens": 512,
  "stream": false
}
```

Response (200):

```json
{
  "query": "What are transformers?",
  "answer": "...",
  "sources": [{"index":1,"content":"...","source":"paper.pdf","score":0.85}],
  "metadata": {"retrieval_time_ms":150,"generation_time_ms":850}
}
```

Streaming: use `stream: true` to receive SSE/streamed tokens (if enabled).

---

## GET /health

Check system health and component status.

Response (200):

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {"ollama":"connected","vector_store":"ready","cache":"enabled"}
}
```

---

## POST /evaluate/retrieval

Evaluate retrieval performance on your dataset.

Request (JSON):

```json
{ "method": "hybrid", "top_k": 5, "num_queries": 20 }
```

Response (200):

```json
{ "method":"hybrid","top_k":5,"metrics":{"precision_at_k":0.85,"recall_at_k":0.78},"total_queries":20 }
```

---

## Error responses

400 Bad Request:

```json
{ "detail": "Query cannot be empty", "error_code": "VALIDATION_ERROR" }
```

500 Internal Server Error:

```json
{ "detail": "An unexpected error occurred", "error_code": "INTERNAL_ERROR" }
```

---

See [GETTING_STARTED.md](GETTING_STARTED.md) for setup and [PARAMETER_TUNING.md](PARAMETER_TUNING.md) for evaluation workflows.
