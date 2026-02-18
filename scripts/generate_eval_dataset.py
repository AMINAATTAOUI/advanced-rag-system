"""
Generate an evaluation dataset from the actual indexed documents.

Reads the ChromaDB index, picks representative chunks from distinct papers,
and builds eval_dataset.json with retrieval + generation questions.
"""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.vector_store import VectorStore
from src.config import settings


def main():
    vs = VectorStore()

    # Collect all unique papers with representative chunks
    r = vs.collection.get(limit=5000, include=['metadatas', 'documents'])
    papers = {}
    for i, meta in enumerate(r['metadatas']):
        src = meta.get('source', '').replace('\\', '/').split('/')[-1]
        if not src:
            continue
        title = meta.get('title', 'Unknown')
        content = r['documents'][i]

        if src not in papers:
            papers[src] = {
                'title': title,
                'chunks': [],
            }
        # Keep a few chunks per paper for keyword extraction
        if len(papers[src]['chunks']) < 3:
            papers[src]['chunks'].append(content[:500])

    print(f"Found {len(papers)} unique papers in the index")

    # Pick 10 papers with the most content
    sorted_papers = sorted(papers.items(), key=lambda x: len(x[1]['chunks']), reverse=True)
    selected = sorted_papers[:10]

    # Templates for generating questions
    question_templates = [
        ("According to the paper '{title}', what is the main contribution or approach proposed?",
         "easy", "retrieval"),
        ("What are key findings or results reported in '{title}'?",
         "medium", "retrieval"),
        ("What is the methodology described in the paper '{title}'?",
         "medium", "generation"),
        ("Based on '{title}', what problem does the paper address?",
         "easy", "generation"),
    ]

    eval_data = []
    qid = 1
    for filename, info in selected:
        title = info['title']
        chunks = info['chunks']

        # Extract keywords from content
        all_text = ' '.join(chunks).lower()
        words = all_text.split()
        # simple keyword extraction: unique words > 5 chars, not stop words
        stop = {'which', 'where', 'about', 'these', 'their', 'there', 'would',
                'could', 'should', 'other', 'using', 'based', 'paper',
                'model', 'results', 'through', 'method'}
        keywords = []
        seen = set()
        for w in words:
            w_clean = w.strip('.,;:()[]{}"\'-')
            if len(w_clean) > 5 and w_clean.isalpha() and w_clean not in stop and w_clean not in seen:
                seen.add(w_clean)
                keywords.append(w_clean)
            if len(keywords) >= 5:
                break

        # Create 2 questions per paper (1 retrieval + 1 generation)
        for template, difficulty, eval_type in question_templates[:2]:
            question = template.format(title=title[:80])
            eval_data.append({
                "id": f"q{qid:02d}",
                "eval_type": eval_type,
                "question": question,
                "expected_answer": f"The paper discusses {title[:100]}.",
                "relevant_docs": [filename],
                "gold_evidence": {
                    "expected_keywords": keywords[:3]
                },
                "difficulty": difficulty
            })
            qid += 1

    # Save
    out_path = os.path.join(settings.test_data_path, 'eval_dataset.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(eval_data)} eval questions -> {out_path}")
    for q in eval_data[:5]:
        print(f"  {q['id']} [{q['eval_type']}] {q['question'][:70]}...")


if __name__ == '__main__':
    main()
