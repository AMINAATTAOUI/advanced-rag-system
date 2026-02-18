"""
Document processor using LangChain text splitters.

LangChain Components Used:
- langchain_text_splitters.RecursiveCharacterTextSplitter  → Token-aware semantic chunking
  with .from_tiktoken_encoder() for precise token-based splitting
- langchain_core.documents.Document                        → Standard document schema

Replaces the previous custom tiktoken-based chunking with LangChain's
production-grade text splitter that handles sentence boundaries, overlap,
and token counting natively.
"""

import re
from typing import List, Dict
from dataclasses import dataclass

# ── LangChain Text Splitter ─────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tiktoken

from src.config import settings
from src.utils.logger import log


@dataclass
class Chunk:
    """Represents a text chunk with metadata (kept for backward compatibility)."""
    content: str
    metadata: Dict
    chunk_id: str
    start_index: int
    end_index: int
    token_count: int


class DocumentProcessor:
    """
    Process and chunk documents using LangChain's RecursiveCharacterTextSplitter.

    Uses RecursiveCharacterTextSplitter.from_tiktoken_encoder() for:
    - Token-based chunk sizing (not character-based)
    - Recursive splitting at semantic boundaries (\\n\\n → \\n → . → space)
    - Configurable overlap in tokens
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = 100,
        respect_sentence_boundary: bool = True
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

        # ── LangChain text splitter with tiktoken token counting ─────
        # Separators define the hierarchy of semantic boundaries:
        #   paragraph → line → sentence → word
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            strip_whitespace=True,
        )

        # Encoder for token counting in stats
        self._encoder = tiktoken.get_encoding("cl100k_base")

        log.info(
            f"DocumentProcessor initialized (LangChain RecursiveCharacterTextSplitter): "
            f"chunk_size={self.chunk_size} tokens, overlap={self.chunk_overlap} tokens, "
            f"semantic_boundaries={'enabled' if respect_sentence_boundary else 'disabled'}"
        )

    # ── process a single document ────────────────────────────────────
    def process_document(self, document: Dict) -> List[Chunk]:
        """
        Process one document into chunks using LangChain's text splitter.
        
        Accepts either:
          1. A dict with 'content' + 'metadata' keys (legacy format)
          2. A dict with 'lc_documents' key (LangChain Documents from loader)
        """
        metadata = document.get("metadata", {})

        # If we already have LangChain Documents, split them directly
        lc_docs = document.get("lc_documents")
        if lc_docs:
            split_docs: List[Document] = self.text_splitter.split_documents(lc_docs)
        else:
            content = document.get("content", "")
            if not content:
                log.warning("Empty document content")
                return []
            content = self._normalize_text(content)
            lc_doc = Document(page_content=content, metadata=metadata)
            split_docs = self.text_splitter.split_documents([lc_doc])

        # Convert LangChain Documents → Chunk dataclass (for backward compatibility)
        chunks = []
        for i, doc in enumerate(split_docs):
            text = doc.page_content
            if len(text) < self.min_chunk_size:
                continue

            token_count = len(self._encoder.encode(text))
            source = doc.metadata.get("source", metadata.get("source", "unknown"))

            chunk = Chunk(
                content=text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "chunk_size_tokens": token_count,
                    "chunk_size_chars": len(text),
                },
                chunk_id=f"{source}_{i}",
                start_index=0,
                end_index=len(text),
                token_count=token_count,
            )
            chunks.append(chunk)

        log.debug(
            f"Created {len(chunks)} chunks from document: "
            f"{metadata.get('source', 'unknown')}"
        )
        return chunks

    # ── process multiple documents ───────────────────────────────────
    def process_documents(self, documents: List[Dict]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        failed_docs = 0

        for doc in documents:
            try:
                chunks = self.process_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                failed_docs += 1
                name = doc.get("metadata", {}).get("source", "unknown")
                log.error(f"Error processing document {name}: {e}")

        log.info(
            f"Processed {len(documents)} documents into {len(all_chunks)} chunks "
            f"({failed_docs} failed)"
        )
        return all_chunks

    # ── convenience: split LangChain Documents directly ──────────────
    def split_langchain_documents(self, lc_docs: List[Document]) -> List[Document]:
        """
        Split a list of LangChain Documents using the configured text splitter.
        
        This is the pure-LangChain path:
            docs  = loader.load_as_langchain_documents("./data/raw")
            chunks = processor.split_langchain_documents(docs)
            Chroma.from_documents(chunks, embedding)
        """
        split = self.text_splitter.split_documents(lc_docs)
        log.info(f"Split {len(lc_docs)} documents into {len(split)} chunks")
        return split

    # ── text normalisation ───────────────────────────────────────────
    def _normalize_text(self, text: str) -> str:
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        text = re.sub(r'\n(?!\n)', ' ', text)
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        return text.strip()

    # ── stats ────────────────────────────────────────────────────────
    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict:
        if not chunks:
            return {
                "total_chunks": 0,
                "target_chunk_size_tokens": self.chunk_size,
                "target_overlap_tokens": self.chunk_overlap,
            }

        sizes_chars = [len(c.content) for c in chunks]
        sizes_tokens = [c.token_count for c in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size_chars": sum(sizes_chars) / len(chunks),
            "min_chunk_size_chars": min(sizes_chars),
            "max_chunk_size_chars": max(sizes_chars),
            "total_characters": sum(sizes_chars),
            "avg_chunk_size_tokens": sum(sizes_tokens) / len(chunks),
            "min_chunk_size_tokens": min(sizes_tokens),
            "max_chunk_size_tokens": max(sizes_tokens),
            "total_tokens": sum(sizes_tokens),
            "target_chunk_size_tokens": self.chunk_size,
            "target_overlap_tokens": self.chunk_overlap,
            "avg_tokens_per_char": sum(sizes_tokens) / max(sum(sizes_chars), 1),
        }


if __name__ == "__main__":
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)

    sample_doc = {
        "content": "This is a sample document. " * 100,
        "metadata": {"source": "test.txt"},
    }

    chunks = processor.process_document(sample_doc)
    stats = processor.get_chunk_stats(chunks)
    print(f"Chunk stats: {stats}")
