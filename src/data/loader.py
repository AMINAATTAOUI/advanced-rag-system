"""
Document loaders for various file formats.
Supports PDF, TXT, DOCX, and more.
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
from pypdf import PdfReader
from docx import Document
from src.utils.logger import log


class DocumentLoader:
    """Load documents from various file formats."""
    
    SUPPORTED_FORMATS = [".pdf", ".txt", ".docx", ".md"]
    
    def __init__(self):
        """Initialize the document loader."""
        log.info("DocumentLoader initialized")
    
    def load_file(self, file_path: str) -> Dict:
        """
        Load a single file and extract its content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file metadata and content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {extension}")
        
        log.debug(f"Loading file: {file_path}")
        
        # Load based on file type
        if extension == ".pdf":
            content = self._load_pdf(file_path)
        elif extension == ".txt" or extension == ".md":
            content = self._load_text(file_path)
        elif extension == ".docx":
            content = self._load_docx(file_path)
        else:
            raise ValueError(f"Unsupported format: {extension}")
        
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": extension,
            "content": content,
            "metadata": {
                "source": str(file_path),
                "file_size": file_path.stat().st_size,
            }
        }
    
    def load_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories
            file_types: List of file extensions to load (default: all supported)
            
        Returns:
            List of document dictionaries
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        file_types = file_types or self.SUPPORTED_FORMATS
        documents = []
        
        # Get all files
        if recursive:
            files = [f for f in directory.rglob("*") if f.is_file()]
        else:
            files = [f for f in directory.glob("*") if f.is_file()]
        
        # Filter by file type
        files = [f for f in files if f.suffix.lower() in file_types]
        
        log.info(f"Found {len(files)} files to load from {directory}")
        
        # Load each file
        for file_path in files:
            try:
                doc = self.load_file(file_path)
                documents.append(doc)
            except Exception as e:
                log.error(f"Error loading {file_path}: {e}")
                continue
        
        log.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def _load_pdf(self, file_path: Path) -> str:
        """Load content from PDF file."""
        try:
            reader = PdfReader(file_path)
            text = []
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            
            content = "\n\n".join(text)
            log.debug(f"Extracted {len(content)} characters from PDF")
            return content
            
        except Exception as e:
            log.error(f"Error reading PDF {file_path}: {e}")
            raise
    
    def _load_text(self, file_path: Path) -> str:
        """Load content from text file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            log.debug(f"Loaded {len(content)} characters from text file")
            return content
            
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    content = f.read()
                log.debug(f"Loaded {len(content)} characters with latin-1 encoding")
                return content
            except Exception as e:
                log.error(f"Error reading text file {file_path}: {e}")
                raise
    
    def _load_docx(self, file_path: Path) -> str:
        """Load content from DOCX file."""
        try:
            doc = Document(file_path)
            text = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text)
            
            content = "\n\n".join(text)
            log.debug(f"Extracted {len(content)} characters from DOCX")
            return content
            
        except Exception as e:
            log.error(f"Error reading DOCX {file_path}: {e}")
            raise
    
    def get_document_stats(self, documents: List[Dict]) -> Dict:
        """
        Get statistics about loaded documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not documents:
            return {"total_documents": 0}
        
        total_chars = sum(len(doc["content"]) for doc in documents)
        file_types = {}
        
        for doc in documents:
            file_type = doc["file_type"]
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "avg_characters_per_doc": total_chars // len(documents),
            "file_types": file_types
        }


if __name__ == "__main__":
    # Example usage
    loader = DocumentLoader()
    
    # Load a single file
    # doc = loader.load_file("path/to/file.pdf")
    
    # Load directory
    # docs = loader.load_directory("./data/raw", recursive=True)
    # stats = loader.get_document_stats(docs)
    # print(f"Document stats: {stats}")
