#!/usr/bin/env python3
"""
Generation Quality Evaluation

Evaluates the quality of generated answers using retrieved context.
Measures faithfulness, relevance, correctness, and hallucination detection.

Usage:
    python tests/test_evaluate_generation.py --method hybrid --num_queries 10
"""

import sys
import os
import json
import time
import argparse
import warnings
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", message="Failed to send telemetry")
warnings.filterwarnings("ignore", message="resume_download")

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm import LLMClient
from src.generation.chain import RAGChain
from src.utils.logger import log


@dataclass
class GenerationResult:
    """Result of a single generation operation."""
    query_id: str
    question: str
    answer: str
    context: List[str]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float


@dataclass
class GenerationMetrics:
    """Evaluation metrics for generation quality."""
    faithfulness: float  # Answer grounded in context
    answer_relevance: float  # Answer addresses question
    context_precision: float  # Retrieved context is relevant
    hallucination_rate: float  # Made-up information rate
    avg_retrieval_time_ms: float
    avg_generation_time_ms: float
    avg_total_time_ms: float
    total_queries: int
    failed_queries: int


class GenerationEvaluator:
    """Evaluates generation quality using RAG pipeline."""

    def __init__(self, method: str = 'hybrid', top_k: int = 5):
        """
        Initialize the generation evaluator.

        Args:
            method: Retrieval method ('dense', 'sparse', 'hybrid')
            top_k: Number of documents to retrieve
        """
        self.method = method
        self.top_k = top_k
        
        # Load evaluation dataset
        self.eval_data = self._load_eval_dataset()
        
        # Initialize components
        self._init_components()
        
        log.info(f"GenerationEvaluator ready: {method}, top_k={top_k}")

    def _load_eval_dataset(self) -> List[Dict]:
        """Load evaluation dataset."""
        eval_path = os.path.join(settings.test_data_path, 'eval_dataset.json')
        if not os.path.exists(eval_path):
            raise FileNotFoundError(f"Evaluation dataset not found: {eval_path}")

        encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'latin-1']
        for enc in encodings:
            try:
                with open(eval_path, 'r', encoding=enc) as f:
                    data = json.load(f)
                log.debug(f"Loaded {len(data)} evaluation queries")
                return data
            except Exception:
                continue
        raise ValueError("Could not load evaluation dataset")

    def _init_components(self):
        """Initialize RAG components."""
        log.debug("Initializing RAG components...")
        
        # Initialize retrieval components
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        
        if self.method in ['sparse', 'hybrid']:
            self.bm25_retriever = BM25Retriever()
            if not self.bm25_retriever.load_index():
                log.warning("BM25 index not found. Building from vector store...")
                all_chunks = self._get_all_chunks()
                if all_chunks:
                    self.bm25_retriever.build_index(all_chunks)
                    self.bm25_retriever.save_index()
        
        if self.method == 'hybrid':
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                bm25_retriever=self.bm25_retriever,
                fusion_strategy='adaptive',
                dense_weight=0.7,
                sparse_weight=0.3
            )
        elif self.method == 'dense':
            self.retriever = self.vector_store
        elif self.method == 'sparse':
            self.retriever = self.bm25_retriever
        
        # Initialize LLM
        self.llm_client = LLMClient()
        
        # Initialize RAG chain
        self.rag_chain = RAGChain(
            retriever=self.retriever,
            llm_client=self.llm_client
        )
        
        log.debug("RAG components ready")

    def _get_all_chunks(self) -> List:
        """Get all chunks from vector store."""
        from src.data.processor import Chunk
        try:
            results = self.vector_store.collection.get(
                limit=10000,
                include=['documents', 'metadatas', 'ids']
            )
            chunks = []
            for i, doc_id in enumerate(results['ids']):
                chunk = Chunk(
                    content=results['documents'][i],
                    metadata=results['metadatas'][i],
                    chunk_id=doc_id,
                    start_index=0,
                    end_index=len(results['documents'][i]),
                    token_count=len(results['documents'][i].split())
                )
                chunks.append(chunk)
            return chunks
        except Exception as e:
            log.error(f"Error getting chunks: {e}")
            return []

    def _generate_answer(self, question: str) -> Tuple[str, List[str], float, float]:
        """
        Generate answer using RAG pipeline.

        Returns:
            Tuple of (answer, context_chunks, retrieval_time_ms, generation_time_ms)
        """
        # Retrieval
        retrieval_start = time.time()
        
        if self.method == 'hybrid':
            retrieved = self.retriever.retrieve(question, top_k=self.top_k, use_reranking=True)
        elif self.method == 'dense':
            query_embedding = self.embedding_model.encode([question])[0]
            retrieved = self.retriever.search_by_embedding(query_embedding, self.top_k)
        elif self.method == 'sparse':
            retrieved = self.retriever.search(question, self.top_k)
        
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # Extract context
        context_chunks = [doc['content'] for doc in retrieved]
        context = "\n\n".join(context_chunks)
        
        # Generation
        generation_start = time.time()
        
        # Use RAG chain to generate answer
        try:
            response = self.rag_chain.query(
                question,
                top_k=self.top_k,
                use_reranking=True,
                include_sources=True,
                system_prompt_type='concise'
            )
            answer = response.get('answer', '')
        except Exception as e:
            log.error(f"Generation failed: {e}")
            answer = ""
        
        generation_time = (time.time() - generation_start) * 1000
        
        return answer, context_chunks, retrieval_time, generation_time

    def _calculate_faithfulness(self, answer: str, context_chunks: List[str]) -> float:
        """
        Calculate faithfulness: Fraction of answer sentences supported by context.
        
        Improved method:
        - Robust sentence tokenization (handles abbreviations)
        - Word tokenization + lemmatization (morphology normalization)
        - Whole-word matching (not substring)
        - Stopword filtering
        
        Returns: 0.0 to 1.0 (fraction of supported sentences)
        """
        if not answer or not context_chunks:
            return 0.0
        
        try:
            import nltk
            from nltk.tokenize import sent_tokenize, word_tokenize
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            # Download required NLTK data (first run only; cached after)
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))
            
            # Normalize context: tokenize, lemmatize, filter stopwords
            full_context = " ".join(context_chunks).lower()
            context_lemmas = set()
            
            for word in word_tokenize(full_context):
                if word.isalnum() and len(word) > 2:
                    lemma = lemmatizer.lemmatize(word)
                    if lemma not in stop_words:
                        context_lemmas.add(lemma)
            
            if not context_lemmas:
                return 0.0
            
            # Split answer into sentences
            answer_lower = answer.lower()
            sentences = sent_tokenize(answer_lower)
            
            if not sentences:
                return 0.0
            
            supported_sentences = 0
            
            # Evaluate each sentence
            for sentence in sentences:
                # Tokenize, lemmatize, filter stopwords
                sentence_lemmas = []
                for word in word_tokenize(sentence):
                    if word.isalnum() and len(word) > 2:
                        lemma = lemmatizer.lemmatize(word)
                        if lemma not in stop_words:
                            sentence_lemmas.append(lemma)
                
                if not sentence_lemmas:
                    # Empty sentence after filtering; count as supported
                    supported_sentences += 1
                    continue
                
                # Count matching lemmas
                matching_lemmas = sum(1 for lemma in sentence_lemmas if lemma in context_lemmas)
                
                # Support if >50% of sentence lemmas match context
                if matching_lemmas / len(sentence_lemmas) > 0.5:
                    supported_sentences += 1
            
            faithfulness = supported_sentences / len(sentences)
            return faithfulness
        
        except Exception as e:
            log.error(f"Faithfulness calculation error (falling back to 0.0): {e}")
            return 0.0

    def _calculate_answer_relevance(self, answer: str, question: str) -> float:
        """
        Calculate answer relevance: How well the answer addresses the question.
        
        Uses keyword overlap between question and answer.
        """
        if not answer or not question:
            return 0.0
        
        # Extract key words from question (remove common words)
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'the', 'a', 'an'}
        question_words = set(w.lower() for w in question.split() if w.lower() not in stop_words and len(w) > 2)
        
        if not question_words:
            return 0.5  # Neutral score if no key words
        
        answer_lower = answer.lower()
        
        # Check how many question keywords appear in answer
        words_in_answer = sum(1 for w in question_words if w in answer_lower)
        
        relevance = words_in_answer / len(question_words)
        return min(relevance, 1.0)

    def _calculate_context_precision(self, context_chunks: List[str], question: str) -> float:
        """
        Calculate context precision: How relevant is the retrieved context.
        
        Uses keyword overlap between question and context.
        """
        if not context_chunks or not question:
            return 0.0
        
        # Extract key words from question
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'the', 'a', 'an'}
        question_words = set(w.lower() for w in question.split() if w.lower() not in stop_words and len(w) > 2)
        
        if not question_words:
            return 0.5
        
        # Check each context chunk
        relevant_chunks = 0
        
        for chunk in context_chunks:
            chunk_lower = chunk.lower()
            words_in_chunk = sum(1 for w in question_words if w in chunk_lower)
            
            if words_in_chunk / len(question_words) > 0.3:  # 30% threshold
                relevant_chunks += 1
        
        precision = relevant_chunks / len(context_chunks)
        return precision

    def _detect_hallucination(self, answer: str, context_chunks: List[str]) -> float:
        """
        Detect hallucination: Information in answer not present in context.
        
        Returns hallucination rate (0 = no hallucination, 1 = high hallucination).
        """
        if not answer or not context_chunks:
            return 0.0
        
        # Inverse of faithfulness is a simple hallucination metric
        faithfulness = self._calculate_faithfulness(answer, context_chunks)
        hallucination_rate = 1.0 - faithfulness
        
        return hallucination_rate

    def evaluate(self, num_queries: int = None) -> GenerationMetrics:
        """
        Run generation evaluation.

        Args:
            num_queries: Number of queries to evaluate (None = all)

        Returns:
            GenerationMetrics object
        """
        # Filter generation-type queries
        generation_queries = [q for q in self.eval_data if q.get('eval_type') == 'generation']
        
        if not generation_queries:
            log.warning("No generation queries found. Using retrieval queries instead.")
            generation_queries = [q for q in self.eval_data if q.get('eval_type') == 'retrieval']
        
        if num_queries:
            generation_queries = generation_queries[:num_queries]
        
        log.info(f"Evaluating {len(generation_queries)} queries...")

        results = []
        failed = 0
        
        faithfulness_scores = []
        relevance_scores = []
        precision_scores = []
        hallucination_scores = []
        retrieval_times = []
        generation_times = []
        total_times = []

        for i, query_data in enumerate(generation_queries):
            question = query_data['question']
            
            try:
                # Generate answer
                start_time = time.time()
                answer, context_chunks, retrieval_time, generation_time = self._generate_answer(question)
                total_time = (time.time() - start_time) * 1000
                
                # Calculate metrics
                faithfulness = self._calculate_faithfulness(answer, context_chunks)
                relevance = self._calculate_answer_relevance(answer, question)
                precision = self._calculate_context_precision(context_chunks, question)
                hallucination = self._detect_hallucination(answer, context_chunks)
                
                # Store scores
                faithfulness_scores.append(faithfulness)
                relevance_scores.append(relevance)
                precision_scores.append(precision)
                hallucination_scores.append(hallucination)
                retrieval_times.append(retrieval_time)
                generation_times.append(generation_time)
                total_times.append(total_time)
                
                log.debug(
                    f"Q{i+1}/{len(generation_queries)}: "
                    f"F={faithfulness:.2f}, R={relevance:.2f}, "
                    f"Time={total_time:.0f}ms"
                )
                
                results.append(GenerationResult(
                    query_id=query_data['id'],
                    question=question,
                    answer=answer,
                    context=context_chunks,
                    retrieval_time_ms=retrieval_time,
                    generation_time_ms=generation_time,
                    total_time_ms=total_time
                ))
                
            except Exception as e:
                failed += 1
                log.error(f"Q{i+1} failed: {str(e)[:50]}")

        # Calculate aggregate metrics
        metrics = GenerationMetrics(
            faithfulness=np.mean(faithfulness_scores) if faithfulness_scores else 0.0,
            answer_relevance=np.mean(relevance_scores) if relevance_scores else 0.0,
            context_precision=np.mean(precision_scores) if precision_scores else 0.0,
            hallucination_rate=np.mean(hallucination_scores) if hallucination_scores else 0.0,
            avg_retrieval_time_ms=np.mean(retrieval_times) if retrieval_times else 0.0,
            avg_generation_time_ms=np.mean(generation_times) if generation_times else 0.0,
            avg_total_time_ms=np.mean(total_times) if total_times else 0.0,
            total_queries=len(results),
            failed_queries=failed
        )

        log.info(
            f"✓ Evaluation complete: {len(results)} queries | "
            f"Faithfulness={metrics.faithfulness:.3f}, "
            f"Relevance={metrics.answer_relevance:.3f}, "
            f"Hallucination={metrics.hallucination_rate:.3f}"
        )

        return metrics, results

    def print_results(self, metrics: GenerationMetrics, results: List[GenerationResult], show_examples: int = 3):
        """Print evaluation results."""
        print("\n" + "="*70)
        print(f"  GENERATION QUALITY EVALUATION - {self.method.upper()}")
        print("="*70)
        print(f"Method: {self.method} | Top-K: {self.top_k}")
        print(f"Queries: {metrics.total_queries}" + (f" | Failed: {metrics.failed_queries}" if metrics.failed_queries > 0 else ""))
        print()
        
        print("Quality Metrics:")
        print(f"  Faithfulness: {metrics.faithfulness:.3f} (answer grounded in context)")
        print(f"  Answer Relevance: {metrics.answer_relevance:.3f} (addresses question)")
        print(f"  Context Precision: {metrics.context_precision:.3f} (retrieved context quality)")
        print(f"  Hallucination Rate: {metrics.hallucination_rate:.3f} (lower is better)")
        print()
        
        print("Performance:")
        print(f"  Retrieval: {metrics.avg_retrieval_time_ms:.0f}ms")
        print(f"  Generation: {metrics.avg_generation_time_ms:.0f}ms")
        print(f"  Total: {metrics.avg_total_time_ms:.0f}ms")
        print()
        
        # Assessment
        print("Assessment:")
        print(f"  Faithfulness: {self._assess_faithfulness(metrics.faithfulness)}")
        print(f"  Relevance: {self._assess_relevance(metrics.answer_relevance)}")
        print(f"  Hallucination: {self._assess_hallucination(metrics.hallucination_rate)}")
        print(f"  Latency: {self._assess_latency(metrics.avg_total_time_ms)}")
        print()
        print(self._assess_overall(metrics))
        
        # # Show examples
        # if show_examples > 0 and results:
        #     print("\n" + "="*70)
        #     print(f"  EXAMPLE GENERATIONS (showing {min(show_examples, len(results))})")
        #     print("="*70)
            
        #     for i, result in enumerate(results[:show_examples]):
        #         print(f"\nExample {i+1}:")
        #         print(f"Question: {result.question}")
        #         print(f"Answer: {result.answer[:200]}..." if len(result.answer) > 200 else f"Answer: {result.answer}")
        #         print(f"Time: {result.total_time_ms:.0f}ms")

    def _assess_faithfulness(self, f): return "✓ EXCELLENT" if f >= 0.8 else "~ GOOD" if f >= 0.6 else "⚠ FAIR" if f >= 0.4 else "✗ POOR"
    def _assess_relevance(self, r): return "✓ EXCELLENT" if r >= 0.8 else "~ GOOD" if r >= 0.6 else "⚠ FAIR" if r >= 0.4 else "✗ POOR"
    def _assess_hallucination(self, h): return "✓ EXCELLENT" if h <= 0.2 else "~ GOOD" if h <= 0.4 else "⚠ FAIR" if h <= 0.6 else "✗ POOR"
    def _assess_latency(self, l): return "✓ FAST" if l <= 3000 else "~ ACCEPTABLE" if l <= 5000 else "⚠ SLOW" if l <= 10000 else "✗ VERY SLOW"

    def _assess_overall(self, m):
        good = sum([
            m.faithfulness >= 0.6,
            m.answer_relevance >= 0.6,
            m.hallucination_rate <= 0.4,
            m.avg_total_time_ms <= 5000
        ])



def main():
    parser = argparse.ArgumentParser(description="Evaluate generation quality")
    parser.add_argument('--method', choices=['dense', 'sparse', 'hybrid'], default='hybrid')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_queries', type=int, default=None, help="Number of queries to evaluate (default: all)")
    parser.add_argument('--show_examples', type=int, default=3, help="Number of example generations to show")
    args = parser.parse_args()

    try:
        evaluator = GenerationEvaluator(method=args.method, top_k=args.top_k)
        metrics, results = evaluator.evaluate(num_queries=args.num_queries)
        evaluator.print_results(metrics, results, show_examples=args.show_examples)
    except Exception as e:
        log.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
