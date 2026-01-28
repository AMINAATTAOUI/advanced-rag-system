"""
Document processor for chunking and preprocessing text.
Implements production-grade token-based chunking with semantic preservation.

Key Features:
- Token-based chunking (not character-based) using tiktoken
- Semantic boundary preservation (respects sentences and words)
- Accurate position tracking in original text
- Configurable overlap with validation
- Comprehensive error handling and logging
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import tiktoken
from src.config import settings
from src.utils.logger import log


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    content: str
    metadata: Dict
    chunk_id: str
    start_index: int
    end_index: int
    token_count: int  # Actual token count for this chunk


class DocumentProcessor:
    """
    Process and chunk documents for RAG using semantic splitting.

    Uses tiktoken for precise token counting and semantic boundary preservation.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = 100,
        respect_sentence_boundary: bool = True
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Target size for each chunk (in TOKENS, not characters)
            chunk_overlap: Overlap between chunks (in TOKENS)
            min_chunk_size: Minimum chunk size to keep (in characters)
            respect_sentence_boundary: If True, try to end chunks at sentence boundaries
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sentence_boundary = respect_sentence_boundary
        
        # Validate configuration
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        # Initialize tiktoken encoder (cl100k_base is used by GPT-4)
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
            log.info(
                f"DocumentProcessor initialized: chunk_size={self.chunk_size} tokens, "
                f"overlap={self.chunk_overlap} tokens, "
                f"semantic_boundaries={'enabled' if respect_sentence_boundary else 'disabled'}"
            )
        except Exception as e:
            log.error(f"Failed to initialize tiktoken encoder: {e}")
            raise

    def process_document(self, document: Dict) -> List[Chunk]:
        """
        Process a document into semantically-aware chunks.

        Args:
            document: Document dictionary with 'content' and optional 'metadata'

        Returns:
            List of Chunk objects with accurate token counts and positions
        """
        content = document.get("content", "")
        if not content:
            log.warning("Empty document content")
            return []
            
        metadata = document.get("metadata", {})

        # Normalize text (light cleaning to preserve semantic structure)
        normalized_content = self._normalize_text(content)

        # Create chunks with semantic boundaries
        chunks = self._create_semantic_chunks(normalized_content, metadata)

        log.debug(
            f"Created {len(chunks)} chunks from document: "
            f"{metadata.get('source', 'unknown')}"
        )

        return chunks

    def process_documents(self, documents: List[Dict]) -> List[Chunk]:
        """
        Process multiple documents into chunks.

        Args:
            documents: List of document dictionaries

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        failed_docs = 0

        for doc in documents:
            try:
                chunks = self.process_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                failed_docs += 1
                doc_name = doc.get("metadata", {}).get("source", "unknown")
                log.error(f"Error processing document {doc_name}: {e}")
                continue

        log.info(
            f"Processed {len(documents)} documents into {len(all_chunks)} chunks "
            f"({failed_docs} failed)"
        )

        return all_chunks

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text while preserving semantic structure.
        
        This is intentionally light - we want to preserve sentence boundaries,
        paragraph structure, and semantic meaning.

        Args:
            text: Raw text

        Returns:
            Normalized text
        """
        # Normalize whitespace (but preserve paragraph breaks)
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n\n+', '\n\n', text)  # Multiple newlines to double newline
        text = re.sub(r'\n(?!\n)', ' ', text)  # Single newlines to space
        
        # Remove control characters but keep basic punctuation
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def _create_semantic_chunks(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Create chunks with semantic boundary preservation.

        Args:
            text: Normalized text
            metadata: Document metadata

        Returns:
            List of Chunk objects
        """
        if not text:
            return []
        
        # Encode entire text to tokens
        tokens = self.encoder.encode(text)
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return []
        
        chunks = []
        chunk_index = 0
        
        # Calculate step size (how much to advance for each chunk)
        step_size = self.chunk_size - self.chunk_overlap
        
        # Ensure step size is at least 1
        if step_size < 1:
            step_size = 1
            log.warning(
                f"Step size too small, adjusted to 1. "
                f"Consider reducing overlap or increasing chunk size."
            )
        
        # Create overlapping chunks
        position = 0
        while position < total_tokens:
            # Get token window
            end_position = min(position + self.chunk_size, total_tokens)
            chunk_tokens = tokens[position:end_position]
            
            # Skip if chunk is too small
            if len(chunk_tokens) < self.min_chunk_size // 4:  # Rough token estimate
                break
            
            # Decode tokens to text
            chunk_text = self.encoder.decode(chunk_tokens)
            
            # Adjust boundaries for semantic preservation if enabled
            if self.respect_sentence_boundary:
                # Adjust start (skip for first chunk)
                if position > 0:
                    chunk_text = self._adjust_chunk_start(chunk_text)
                # Always adjust end to finish at sentence boundary
                chunk_text = self._adjust_chunk_end(chunk_text)
            
            # Skip if chunk is too small after adjustment
            if len(chunk_text) < self.min_chunk_size:
                position += step_size
                continue
            
            # Find actual position in original text
            start_char, end_char = self._find_chunk_position(text, chunk_text, position)
            
            # Re-encode adjusted chunk to get accurate token count
            actual_tokens = self.encoder.encode(chunk_text)
            
            # Create chunk object
            chunk = Chunk(
                content=chunk_text,
                metadata={
                    **metadata,
                    "chunk_index": chunk_index,
                    "chunk_size_tokens": len(actual_tokens),
                    "chunk_size_chars": len(chunk_text),
                    "total_tokens": total_tokens,
                    "position_tokens": position
                },
                chunk_id=f"{metadata.get('source', 'unknown')}_{chunk_index}",
                start_index=start_char,
                end_index=end_char,
                token_count=len(actual_tokens)
            )
            
            chunks.append(chunk)
            chunk_index += 1
            
            # Move to next position
            position += step_size
            
            # Break if we've processed the entire text
            if end_position >= total_tokens:
                break
        
        # Validate overlap if we have multiple chunks
        if len(chunks) > 1:
            self._validate_overlap(chunks)
        
        return chunks

    def _adjust_chunk_start(self, chunk_text: str) -> str:
        """
        Adjust chunk start to begin at a sentence or word boundary.

        Args:
            chunk_text: Raw chunk text

        Returns:
            Adjusted chunk text
        """
        # Try to find sentence boundary (. ! ? followed by space and capital letter)
        sentence_pattern = r'[.!?]\s+[A-Z]'
        match = re.search(sentence_pattern, chunk_text)
        
        if match:
            # Start after the sentence boundary
            start_pos = match.start() + 2  # After punctuation and space
            return chunk_text[start_pos:].strip()
        
        # If no sentence boundary, try to start at a word boundary
        # Find first space after some initial characters
        if len(chunk_text) > 20:
            space_pos = chunk_text.find(' ', 10)
            if space_pos > 0:
                return chunk_text[space_pos + 1:].strip()
        
        # If all else fails, return as is
        return chunk_text.strip()

    def _adjust_chunk_end(self, chunk_text: str, max_reduction: int = 100) -> str:
        """
        Adjust chunk end to finish at a sentence boundary.
        
        This prevents chunks from ending mid-sentence, improving semantic
        completeness and context quality.

        Args:
            chunk_text: Raw chunk text
            max_reduction: Maximum characters to trim from end

        Returns:
            Adjusted chunk text ending at sentence boundary
        """
        if len(chunk_text) <= max_reduction:
            # Chunk too short to adjust
            return chunk_text.strip()
        
        # Find last sentence boundary within max_reduction chars from end
        search_start = max(0, len(chunk_text) - max_reduction)
        search_text = chunk_text[search_start:]
        
        # Look for sentence endings (. ! ? : ;)
        sentence_endings = ['.', '!', '?', ':', ';']
        last_boundary = -1
        best_ending = None
        
        for ending in sentence_endings:
            pos = search_text.rfind(ending)
            if pos > last_boundary:
                last_boundary = pos
                best_ending = ending
        
        if last_boundary > 0:
            # Cut at sentence boundary (include the punctuation)
            actual_pos = search_start + last_boundary + 1
            result = chunk_text[:actual_pos].strip()
            
            # Ensure we didn't cut too much
            if len(result) >= self.min_chunk_size:
                return result
        
        # Fallback: cut at last word boundary
        last_space = chunk_text.rfind(' ', len(chunk_text) - max_reduction)
        if last_space > 0 and last_space >= self.min_chunk_size:
            return chunk_text[:last_space].strip()
        
        # Last resort: return as is
        return chunk_text.strip()

    def _find_chunk_position(
        self, 
        full_text: str, 
        chunk_text: str, 
        token_position: int
    ) -> Tuple[int, int]:
        """
        Find the character position of a chunk in the original text.
        
        This provides accurate start_index and end_index for each chunk.

        Args:
            full_text: Complete original text
            chunk_text: Chunk text to find
            token_position: Approximate token position (for optimization)

        Returns:
            Tuple of (start_index, end_index) in characters
        """
        # Estimate character position from token position
        # Average ~4 characters per token for English text
        estimated_char_pos = max(0, token_position * 4 - 100)
        
        # Search for chunk in a window around estimated position
        search_start = max(0, estimated_char_pos)
        search_end = min(len(full_text), estimated_char_pos + len(chunk_text) * 2)
        search_window = full_text[search_start:search_end]
        
        # Find chunk in window
        chunk_start_in_window = search_window.find(chunk_text[:50])  # Use first 50 chars
        
        if chunk_start_in_window >= 0:
            start_index = search_start + chunk_start_in_window
            end_index = start_index + len(chunk_text)
            return start_index, end_index
        
        # Fallback: search entire text (slower but accurate)
        start_index = full_text.find(chunk_text[:50])
        if start_index >= 0:
            end_index = start_index + len(chunk_text)
            return start_index, end_index
        
        # Last resort: use estimated position
        log.warning(f"Could not find exact position for chunk at token {token_position}")
        start_index = estimated_char_pos
        end_index = start_index + len(chunk_text)
        return start_index, end_index

    def _validate_overlap(self, chunks: List[Chunk]) -> None:
        """
        Validate that chunks have proper overlap.
        
        Logs warnings if overlap is significantly different from configured value.

        Args:
            chunks: List of chunks to validate
        """
        if len(chunks) < 2:
            return
        
        overlaps = []
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Check for content overlap
            current_end = current_chunk.content[-100:]  # Last 100 chars
            next_start = next_chunk.content[:100]  # First 100 chars
            
            # Find common substring
            overlap_found = False
            for length in range(min(len(current_end), len(next_start)), 10, -1):
                if current_end[-length:] in next_start:
                    overlap_found = True
                    break
            
            if overlap_found:
                overlaps.append(True)
            else:
                log.debug(
                    f"No content overlap detected between chunks {i} and {i+1}. "
                    f"This may be due to semantic boundary adjustments."
                )
        
        if overlaps:
            overlap_rate = len(overlaps) / (len(chunks) - 1)
            if overlap_rate < 0.5:
                log.warning(
                    f"Low overlap rate: {overlap_rate:.2%}. "
                    f"Consider adjusting chunk_overlap parameter."
                )

    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict:
        """
        Get comprehensive statistics about chunks.

        Args:
            chunks: List of Chunk objects

        Returns:
            Statistics dictionary with token and character metrics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "target_chunk_size_tokens": self.chunk_size,
                "target_overlap_tokens": self.chunk_overlap
            }

        chunk_sizes_chars = [len(chunk.content) for chunk in chunks]
        chunk_sizes_tokens = [chunk.token_count for chunk in chunks]

        stats = {
            "total_chunks": len(chunks),
            # Character statistics
            "avg_chunk_size_chars": sum(chunk_sizes_chars) / len(chunks),
            "min_chunk_size_chars": min(chunk_sizes_chars),
            "max_chunk_size_chars": max(chunk_sizes_chars),
            "total_characters": sum(chunk_sizes_chars),
            # Token statistics
            "avg_chunk_size_tokens": sum(chunk_sizes_tokens) / len(chunks),
            "min_chunk_size_tokens": min(chunk_sizes_tokens),
            "max_chunk_size_tokens": max(chunk_sizes_tokens),
            "total_tokens": sum(chunk_sizes_tokens),
            # Configuration
            "target_chunk_size_tokens": self.chunk_size,
            "target_overlap_tokens": self.chunk_overlap,
            # Efficiency metrics
            "avg_tokens_per_char": sum(chunk_sizes_tokens) / sum(chunk_sizes_chars),
        }

        return stats


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)

    # Example document
    sample_doc = {
        "content": "This is a sample document. " * 100,
        "metadata": {"source": "test.txt"}
    }

    chunks = processor.process_document(sample_doc)
    stats = processor.get_chunk_stats(chunks)
    print(f"Chunk stats: {stats}")
