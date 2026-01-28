"""
Script to download ArXiv papers for the RAG system.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.downloader import ArxivDownloader
from src.utils.logger import log


def main():
    parser = argparse.ArgumentParser(description="Download ArXiv papers")
    parser.add_argument(
        "--num-papers",
        type=int,
        default=100,
        help="Number of papers to download (default: 100)"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["cs.AI", "cs.CL", "cs.LG"],
        help="Research paper categories to download from (default: cs.AI cs.CL cs.LG)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for downloaded papers"
    )
    
    args = parser.parse_args()
    
    log.info("="*60)
    log.info("ArXiv Paper Downloader")
    log.info("="*60)
    log.info(f"Categories: {args.categories}")
    log.info(f"Number of papers: {args.num_papers}")
    
    # Initialize downloader
    downloader = ArxivDownloader(output_dir=args.output_dir)
    
    # Download papers
    try:
        papers = downloader.download_papers(
            categories=args.categories,
            max_results=args.num_papers
        )
        
        # Display stats
        stats = downloader.get_paper_stats()
        log.info("\n" + "="*60)
        log.info("Download Complete!")
        log.info("="*60)
        log.info(f"Total papers downloaded: {stats['total_papers']}")
        log.info(f"Categories breakdown:")
        for cat, count in stats['categories'].items():
            log.info(f"  {cat}: {count}")
        
        if stats['total_papers'] > 0:
            log.info(f"\nDate range:")
            log.info(f"  Earliest: {stats['date_range']['earliest']}")
            log.info(f"  Latest: {stats['date_range']['latest']}")
        
        log.info(f"\nPapers saved to: {downloader.output_dir}")
        log.info("Metadata saved to: papers_metadata.json")
        
    except Exception as e:
        log.error(f"Error downloading papers: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
