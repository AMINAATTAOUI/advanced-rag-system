"""
ArXiv paper downloader for building the RAG dataset.
Downloads papers from specified categories and saves them locally.
"""

import arxiv
import os
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from src.config import settings
from src.utils.logger import log


class ArxivDownloader:
    """Download and manage ArXiv papers for the RAG system."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the ArXiv downloader.
        
        Args:
            output_dir: Directory to save downloaded papers
        """
        self.output_dir = Path(output_dir or settings.data_raw_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"ArXiv downloader initialized. Output: {self.output_dir}")
    
    def download_papers(
        self,
        categories: List[str] = None,
        max_results: int = None,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate
    ) -> List[Dict]:
        """
        Download papers from ArXiv.
        
        Args:
            categories: List of ArXiv categories (e.g., ['cs.AI', 'cs.CL'])
            max_results: Maximum number of papers to download
            sort_by: Sort criterion for results
            
        Returns:
            List of paper metadata dictionaries
        """
        categories = categories or settings.arxiv_categories
        max_results = max_results or settings.num_papers
        
        log.info(f"Downloading {max_results} papers from categories: {categories}")
        
        # Build search query
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])
        
        # Search ArXiv
        search = arxiv.Search(
            query=category_query,
            max_results=max_results,
            sort_by=sort_by
        )
        
        papers_metadata = []
        
        try:
            for paper in tqdm(search.results(), total=max_results, desc="Downloading papers"):
                try:
                    # Create paper metadata
                    metadata = {
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors],
                        "abstract": paper.summary,
                        "published": paper.published.isoformat(),
                        "updated": paper.updated.isoformat(),
                        "categories": paper.categories,
                        "arxiv_id": paper.entry_id.split("/")[-1],
                        "pdf_url": paper.pdf_url,
                    }
                    
                    # Download PDF
                    pdf_filename = f"{metadata['arxiv_id'].replace('/', '_')}.pdf"
                    pdf_path = self.output_dir / pdf_filename
                    
                    if not pdf_path.exists():
                        paper.download_pdf(dirpath=str(self.output_dir), filename=pdf_filename)
                        log.debug(f"Downloaded: {metadata['title']}")
                    else:
                        log.debug(f"Already exists: {metadata['title']}")
                    
                    metadata["local_path"] = str(pdf_path)
                    papers_metadata.append(metadata)
                    
                except Exception as e:
                    log.error(f"Error downloading paper: {e}")
                    continue
        
        except Exception as e:
            log.error(f"Error in search: {e}")
        
        log.info(f"Successfully downloaded {len(papers_metadata)} papers")
        
        # Save metadata
        self._save_metadata(papers_metadata)
        
        return papers_metadata
    
    def _save_metadata(self, papers_metadata: List[Dict]):
        """Save paper metadata to JSON file."""
        import json
        
        metadata_path = self.output_dir / "papers_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(papers_metadata, f, indent=2, ensure_ascii=False)
        
        log.info(f"Metadata saved to {metadata_path}")
    
    def load_metadata(self) -> List[Dict]:
        """Load paper metadata from JSON file."""
        import json
        
        metadata_path = self.output_dir / "papers_metadata.json"
        if not metadata_path.exists():
            log.warning("No metadata file found")
            return []
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_paper_stats(self) -> Dict:
        """Get statistics about downloaded papers."""
        metadata = self.load_metadata()
        
        if not metadata:
            return {"total_papers": 0}
        
        categories = {}
        for paper in metadata:
            for cat in paper.get("categories", []):
                categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_papers": len(metadata),
            "categories": categories,
            "date_range": {
                "earliest": min(p["published"] for p in metadata),
                "latest": max(p["published"] for p in metadata)
            }
        }


if __name__ == "__main__":
    # Example usage
    downloader = ArxivDownloader()
    papers = downloader.download_papers(max_results=10)
    stats = downloader.get_paper_stats()
    print(f"Downloaded papers stats: {stats}")
