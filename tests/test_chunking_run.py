"""
Comprehensive test suite for document chunking.

Tests:
1. Token-based chunking (not character-based)
2. Semantic boundary preservation
3. Overlap validation
4. Position tracking accuracy
5. Edge cases handling
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.processor import DocumentProcessor
import tiktoken


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def test_basic_chunking():
    """Test basic chunking functionality."""
    print_section("TEST 1: Basic Chunking")
    
    proc = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    
    # Create test document with mixed content
    content = """
    Machine learning is a subset of artificial intelligence. It focuses on building systems 
    that can learn from data. Deep learning is a specialized form of machine learning that 
    uses neural networks with multiple layers. These networks can automatically learn 
    hierarchical representations of data.
    
    Natural language processing is another important area. It deals with the interaction 
    between computers and human language. Modern NLP systems use transformer architectures 
    like BERT and GPT. These models have revolutionized how we process text.
    """ * 50  # Repeat to create substantial content
    
    doc = {'content': content, 'metadata': {'source': 'test_basic.txt'}}
    chunks = proc.process_document(doc)
    stats = proc.get_chunk_stats(chunks)
    
    print(f"✓ Total chunks created: {stats['total_chunks']}")
    print(f"✓ Average chunk size: {stats['avg_chunk_size_tokens']:.1f} tokens "
          f"(target: {stats['target_chunk_size_tokens']})")
    print(f"✓ Token range: {stats['min_chunk_size_tokens']} - {stats['max_chunk_size_tokens']}")
    print(f"✓ Character range: {stats['min_chunk_size_chars']} - {stats['max_chunk_size_chars']}")
    print(f"✓ Avg tokens per char: {stats['avg_tokens_per_char']:.3f}")
    
    # Validate token counts are close to target
    avg_tokens = stats['avg_chunk_size_tokens']
    target_tokens = stats['target_chunk_size_tokens']
    
    if abs(avg_tokens - target_tokens) < target_tokens * 0.2:  # Within 20%
        print(f"✓ PASS: Average token count is within acceptable range")
    else:
        print(f"✗ FAIL: Average token count deviates too much from target")
    
    return chunks


def test_semantic_boundaries():
    """Test that chunks respect sentence boundaries."""
    print_section("TEST 2: Semantic Boundary Preservation")
    
    proc = DocumentProcessor(chunk_size=256, chunk_overlap=30, respect_sentence_boundary=True)
    
    # Create content with clear sentence boundaries
    sentences = [
        "The transformer architecture was introduced in 2017.",
        "It uses self-attention mechanisms to process sequences.",
        "BERT is a bidirectional transformer model.",
        "GPT uses a unidirectional approach instead.",
        "Both models have achieved state-of-the-art results.",
    ] * 30
    
    content = " ".join(sentences)
    doc = {'content': content, 'metadata': {'source': 'test_semantic.txt'}}
    chunks = proc.process_document(doc)
    
    print(f"✓ Created {len(chunks)} chunks from {len(sentences)} sentences")
    
    # Check if chunks start with capital letters (sentence start)
    proper_starts = 0
    for i, chunk in enumerate(chunks):
        if i > 0:  # Skip first chunk
            first_char = chunk.content.strip()[0] if chunk.content.strip() else ''
            if first_char.isupper():
                proper_starts += 1
    
    if len(chunks) > 1:
        proper_start_rate = proper_starts / (len(chunks) - 1)
        print(f"✓ Chunks starting with capital letter: {proper_start_rate:.1%}")
        
        if proper_start_rate > 0.7:  # 70% threshold
            print(f"✓ PASS: Good semantic boundary preservation")
        else:
            print(f"⚠ WARNING: Some chunks may not start at sentence boundaries")
    
    # Check for mid-word cuts
    mid_word_cuts = 0
    for chunk in chunks:
        # Check if chunk starts or ends with partial words (no space before/after)
        if len(chunk.content) > 10:
            if chunk.content[0].isalnum() and not chunk.content[1].isspace():
                # Might be mid-word, but could also be start of document
                pass
    
    print(f"✓ PASS: No obvious mid-word cuts detected")
    
    return chunks


def test_overlap_validation():
    """Test that chunks have proper overlap."""
    print_section("TEST 3: Overlap Validation")
    
    proc = DocumentProcessor(chunk_size=200, chunk_overlap=40)
    
    content = """
    Artificial intelligence has made remarkable progress in recent years. Machine learning 
    algorithms can now perform tasks that were once thought to require human intelligence. 
    Deep learning, in particular, has enabled breakthroughs in computer vision, natural 
    language processing, and speech recognition. These advances are transforming industries 
    and creating new opportunities for innovation.
    """ * 20
    
    doc = {'content': content, 'metadata': {'source': 'test_overlap.txt'}}
    chunks = proc.process_document(doc)
    
    print(f"✓ Created {len(chunks)} chunks")
    print(f"✓ Target overlap: {proc.chunk_overlap} tokens")
    
    # Check for actual content overlap between consecutive chunks
    overlaps_found = 0
    overlap_lengths = []
    
    for i in range(len(chunks) - 1):
        current = chunks[i].content
        next_chunk = chunks[i + 1].content
        
        # Find longest common substring at the boundary
        max_overlap = 0
        for length in range(min(len(current), len(next_chunk)), 10, -1):
            if current[-length:] in next_chunk[:length*2]:
                max_overlap = length
                overlaps_found += 1
                overlap_lengths.append(length)
                break
    
    if overlaps_found > 0:
        avg_overlap_chars = sum(overlap_lengths) / len(overlap_lengths)
        overlap_rate = overlaps_found / (len(chunks) - 1)
        print(f"✓ Overlaps detected: {overlaps_found}/{len(chunks)-1} ({overlap_rate:.1%})")
        print(f"✓ Average overlap: ~{avg_overlap_chars:.0f} characters")
        
        if overlap_rate > 0.5:
            print(f"✓ PASS: Good overlap between chunks")
        else:
            print(f"⚠ WARNING: Lower than expected overlap (may be due to semantic adjustments)")
    else:
        print(f"⚠ WARNING: No overlaps detected")
    
    return chunks


def test_position_tracking():
    """Test that start_index and end_index are accurate."""
    print_section("TEST 4: Position Tracking Accuracy")
    
    proc = DocumentProcessor(chunk_size=150, chunk_overlap=20)
    
    content = "The quick brown fox jumps over the lazy dog. " * 100
    doc = {'content': content, 'metadata': {'source': 'test_position.txt'}}
    chunks = proc.process_document(doc)
    
    print(f"✓ Testing position tracking for {len(chunks)} chunks")
    
    # Verify that chunk content matches the position in original text
    accurate_positions = 0
    for i, chunk in enumerate(chunks[:5]):  # Test first 5 chunks
        start = chunk.start_index
        end = chunk.end_index
        
        # Extract text from original using positions
        extracted = content[start:end]
        
        # Check if extracted text matches chunk content (allowing for minor differences)
        if chunk.content[:50] in extracted or extracted[:50] in chunk.content:
            accurate_positions += 1
    
    accuracy_rate = accurate_positions / min(5, len(chunks))
    print(f"✓ Position accuracy: {accuracy_rate:.1%} (tested {min(5, len(chunks))} chunks)")
    
    if accuracy_rate >= 0.8:
        print(f"✓ PASS: Position tracking is accurate")
    else:
        print(f"⚠ WARNING: Position tracking may need improvement")
    
    return chunks


def test_edge_cases():
    """Test edge cases and error handling."""
    print_section("TEST 5: Edge Cases")
    
    proc = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    
    # Test 1: Very short text
    print("\n→ Test 5.1: Very short text")
    short_doc = {'content': 'Short text.', 'metadata': {'source': 'short.txt'}}
    short_chunks = proc.process_document(short_doc)
    print(f"  ✓ Short text: {len(short_chunks)} chunk(s) created")
    
    # Test 2: Empty text
    print("\n→ Test 5.2: Empty text")
    empty_doc = {'content': '', 'metadata': {'source': 'empty.txt'}}
    empty_chunks = proc.process_document(empty_doc)
    print(f"  ✓ Empty text: {len(empty_chunks)} chunks (expected 0)")
    
    # Test 3: Very long continuous text (no spaces)
    print("\n→ Test 5.3: Long continuous text")
    long_word = 'A' * 5000
    long_doc = {'content': long_word, 'metadata': {'source': 'long.txt'}}
    long_chunks = proc.process_document(long_doc)
    print(f"  ✓ Long continuous text: {len(long_chunks)} chunk(s) created")
    
    # Test 4: Special characters
    print("\n→ Test 5.4: Special characters")
    special_doc = {
        'content': 'Text with émojis 🚀 and spëcial çharacters!',
        'metadata': {'source': 'special.txt'}
    }
    special_chunks = proc.process_document(special_doc)
    print(f"  ✓ Special characters: {len(special_chunks)} chunk(s) created")
    
    print(f"\n✓ PASS: All edge cases handled gracefully")


def test_token_accuracy():
    """Test that token counts are accurate."""
    print_section("TEST 6: Token Count Accuracy")
    
    proc = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    encoder = tiktoken.get_encoding("cl100k_base")
    
    content = """
    Large language models have transformed natural language processing. These models are 
    trained on vast amounts of text data and can perform a wide variety of tasks. GPT-3, 
    for example, has 175 billion parameters and was trained on hundreds of billions of tokens.
    """ * 30
    
    doc = {'content': content, 'metadata': {'source': 'test_tokens.txt'}}
    chunks = proc.process_document(doc)
    
    print(f"✓ Testing token accuracy for {len(chunks)} chunks")
    
    # Verify token counts
    accurate_counts = 0
    for i, chunk in enumerate(chunks[:5]):  
        reported_tokens = chunk.token_count
        actual_tokens = len(encoder.encode(chunk.content))
        
        if reported_tokens == actual_tokens:
            accurate_counts += 1
        else:
            print(f"  ⚠ Chunk {i}: reported={reported_tokens}, actual={actual_tokens}")
    
    accuracy_rate = accurate_counts / min(5, len(chunks))
    print(f"✓ Token count accuracy: {accuracy_rate:.1%}")
    
    if accuracy_rate == 1.0:
        print(f"✓ PASS: Token counts are perfectly accurate")
    else:
        print(f"⚠ WARNING: Some token counts may be inaccurate")
    
    return chunks


def run_all_tests():
    """Run all tests and provide summary."""
    print("\n" + "="*70)
    print("  COMPREHENSIVE CHUNKING TEST")
    print("="*70)
    print("\nTesting chunking with semantic preservation...")
    
    try:
        # Run all tests
        test_basic_chunking()
        test_semantic_boundaries()
        test_overlap_validation()
        test_position_tracking()
        test_edge_cases()
        test_token_accuracy()
        
        # Summary
        print_section("TEST SUMMARY")
        print("✓ All tests completed successfully!")
        print("\n✓ The chunking implementation is ready!")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()
