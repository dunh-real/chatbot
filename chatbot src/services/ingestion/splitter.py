import uuid
import shutil
import logging
from pathlib import Path
from typing import List, Dict
from fastapi import UploadFile, HTTPException

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from services.retrieval.qdrant_service import vector_store_service

from .converter import document_converter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".xlsx", ".pptx"}
TARGET_CHUNK_SIZE = 1000

class IngestionService:
    def __init__(self):
        # 1. Embedding Model cho Semantic Chunker
        logger.info("⏳ Loading Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="google/embeddinggemma-300m",
            model_kwargs={'device': 'cuda'}, # 'cpu' nếu không có GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 2. Semantic Splitter (Pure AI Mode)
        self.semantic_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile" 
        )

        # 3. Markdown Splitter (Xác định ranh giới đề mục)
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"),
                ("##", "H2"),
            ],
            strip_headers=False
        )

    def process_hybrid_splitting(self, text: str, tenant_id: str, filename: str) -> List[Dict]:
        """
        Markdown Headers -> Semantic Split (No Recursive)
        """
        header_splits = self.header_splitter.split_text(text)
        final_chunks = []
        
        for doc in header_splits:
            content = doc.page_content
            headers = doc.metadata

            # Nếu đoạn văn quá dài so với mục tiêu, dùng AI để tìm điểm ngắt ý nghĩa
            if len(content) > TARGET_CHUNK_SIZE:
                try:
                    sub_docs = self.semantic_splitter.create_documents([content])
                    for sub in sub_docs:
                        self._add_chunk(final_chunks, sub.page_content, headers, tenant_id, filename, "hybrid_semantic")
                except Exception as e:
                    logger.error(f"Semantic split error: {e}. Keeping original.")
                    self._add_chunk(final_chunks, content, headers, tenant_id, filename, "fallback_original")
            else:
                self._add_chunk(final_chunks, content, headers, tenant_id, filename, "markdown_structure")

        return final_chunks

    def _add_chunk(self, chunks_list, content, headers, tenant_id, filename, method):
            # ... (logic kiểm tra độ dài giữ nguyên)

            enriched_content = self._inject_header_context(content, headers)
            
            # Làm phẳng Metadata (Flatten) ngay tại nguồn
            flat_metadata = {
                "length": len(enriched_content),
                "method": method,
                # Lấy H1, H2, H3 ra ngoài. Nếu không có thì để chuỗi rỗng.
                "h1": headers.get("H1", ""),
                "h2": headers.get("H2", ""),
                "h3": headers.get("H3", "")
            }

            chunks_list.append({
                "chunk_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "src_file": filename,
                "content": enriched_content,
                "metadata": flat_metadata # Metadata đã phẳng, gọn gàng
            })

    def _inject_header_context(self, content: str, metadata: Dict) -> str:
        headers = [metadata[h] for h in ["H1", "H2"] if h in metadata]
        if not headers: return content
        context_str = " > ".join(headers)
        if context_str not in content[:200]:
            return f"**Ngữ cảnh: {context_str}**\n\n{content}"
        return content

# Singleton instance
ingestion_service = IngestionService()

async def process_uploaded_file(file: UploadFile, tenant_id: str) -> List[Dict]:
    request_id = str(uuid.uuid4())
    temp_dir = Path(f"/tmp/ingest_{request_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Validate & Save file
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, "Định dạng không hỗ trợ.")
            
        temp_path = temp_dir / file.filename
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Convert to Markdown (Gọi từ converter.py)
        markdown_text = document_converter.convert(temp_path)
        
        # 3. Chunking
        chunks = ingestion_service.process_hybrid_splitting(markdown_text, tenant_id, file.filename)
        
        logger.info(f"File: {file.filename} -> {len(chunks)} chunks")
        return chunks

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)