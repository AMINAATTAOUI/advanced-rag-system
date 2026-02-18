"""
Document loaders using LangChain's document loader ecosystem.

LangChain Components Used:
- langchain_community.document_loaders.PyPDFLoader     → PDF ingestion (one Document per page)
- langchain_community.document_loaders.TextLoader      → TXT/MD ingestion
- langchain_community.document_loaders.Docx2txtLoader  → DOCX ingestion
- langchain_core.documents.Document                    → Unified document schema
"""

from typing import List, Dict, Optional
from pathlib import Path

# ── LangChain Document Loaders ──────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from src.utils.logger import log


class DocumentLoader:
    """
    Load documents via LangChain document loaders.
    
    Each file format uses its dedicated LangChain loader, producing
    standardised langchain_core.documents.Document objects that plug
    directly into LangChain text-splitters and vector stores.
    """
    
    SUPPORTED_FORMATS = [".pdf", ".txt", ".docx", ".md"]
    
    def __init__(self):
        log.info("DocumentLoader initialized (LangChain PyPDFLoader / TextLoader / Docx2txtLoader)")
    
    # ── internal: select the right LangChain loader ──────────────────
    def _get_loader(self, file_path: Path):
        """
        Return the appropriate LangChain document loader for a file type.
        
        .pdf  → PyPDFLoader   (returns one Document per page)
        .txt  → TextLoader    (returns a single Document)
        .md   → TextLoader
        .docx → Docx2txtLoader
        """
        ext = file_path.suffix.lower()
        
        if ext == ".pdf":
            return PyPDFLoader(str(file_path))
        elif ext in (".txt", ".md"):
            return TextLoader(str(file_path), encoding="utf-8")
        elif ext == ".docx":
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                return Docx2txtLoader(str(file_path))
            except ImportError:
                log.warning("docx2txt not installed – falling back to python-docx")
                return None
        return None
    
    # ── load a single file ───────────────────────────────────────────
    def load_file(self, file_path: str) -> Dict:
        """
        Load a single file through its LangChain loader.
        
        Returns a dict that contains:
          - content          (str)   – concatenated page text
          - metadata         (dict)  – source + file size
          - lc_documents     (List[Document]) – raw LangChain Documents
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {extension}")
        
        log.debug(f"Loading with LangChain loader: {file_path}")
        
        loader = self._get_loader(file_path)
        
        if loader is not None:
            # ── LangChain loader produces List[Document] ────────
            lc_docs: List[Document] = loader.load()
            content = "\n\n".join(doc.page_content for doc in lc_docs)
        else:
            # ── fallback for DOCX without docx2txt ──────────────
            content = self._fallback_load_docx(file_path)
            lc_docs = [Document(page_content=content, metadata={"source": str(file_path)})]
        
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": extension,
            "content": content,
            "metadata": {
                "source": str(file_path),
                "file_size": file_path.stat().st_size,
            },
            # Keep raw LangChain Documents for downstream splitters / vectorstores
            "lc_documents": lc_docs,
        }
    
    def _fallback_load_docx(self, file_path: Path) -> str:
        """Fallback DOCX loader using python-docx when docx2txt is unavailable."""
        from docx import Document as DocxDoc
        doc = DocxDoc(file_path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    
    # ── load a whole directory ───────────────────────────────────────
    def load_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """Load all supported documents from a directory."""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        file_types = file_types or self.SUPPORTED_FORMATS
        documents = []
        
        files = list(directory.rglob("*") if recursive else directory.glob("*"))
        files = [f for f in files if f.is_file() and f.suffix.lower() in file_types]
        
        log.info(f"Found {len(files)} files to load from {directory}")
        
        for fp in files:
            try:
                doc = self.load_file(fp)
                documents.append(doc)
            except Exception as e:
                log.error(f"Error loading {fp}: {e}")
                continue
        
        log.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    # ── convenience: flat list of LangChain Documents ────────────────
    def load_as_langchain_documents(
        self,
        directory: str,
        recursive: bool = True,
        file_types: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load files and return a flat list of LangChain Document objects.
        
        These can be passed directly to:
            text_splitter.split_documents(docs)
            Chroma.from_documents(docs, embedding)
        """
        raw = self.load_directory(directory, recursive, file_types)
        lc_docs = []
        for d in raw:
            lc_docs.extend(d.get("lc_documents", []))
        log.info(f"Returned {len(lc_docs)} LangChain Document objects")
        return lc_docs
    
    # ── stats ────────────────────────────────────────────────────────
    def get_document_stats(self, documents: List[Dict]) -> Dict:
        if not documents:
            return {"total_documents": 0}
        total_chars = sum(len(doc["content"]) for doc in documents)
        file_types: Dict[str, int] = {}
        for doc in documents:
            ft = doc["file_type"]
            file_types[ft] = file_types.get(ft, 0) + 1
        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "avg_characters_per_doc": total_chars // len(documents),
            "file_types": file_types,
        }


if __name__ == "__main__":
    loader = DocumentLoader()
    # docs = loader.load_as_langchain_documents("./data/raw")
    # print(f"Loaded {len(docs)} LangChain Documents")
