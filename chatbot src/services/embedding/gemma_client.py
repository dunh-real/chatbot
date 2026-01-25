import logging
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List 

logger = logging.getLogger(__name__)

class GemmaEmbeddingClient:
    def __init__(self, model_name: str = "google/embeddinggemma-300m"):
        self.model_name = model_name
        # Tự động kiểm tra GPU, CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"⏳ Khởi tạo Gemma Embedding Model trên thiết bị: {self.device}...")
        
        try:
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={
                    'device': self.device,
                    'trust_remote_code': True # Cần thiết cho một số model Gemma trên HF
                },
                encode_kwargs={
                    'normalize_embeddings': True # Chuẩn hóa để tính toán khoảng cách Cosine chính xác
                }
            )
            logger.info("✅ Gemma Embedding Model đã sẵn sàng!")
        except Exception as e:
            logger.error(f"❌ Lỗi khởi tạo Embedding Model: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        Chuyển đổi 1 câu text thành dense vector (list float)
        """
        # encode trả về numpy array, cần convert sang list cho Qdrant
        return self.model.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Chuyển đổi danh sách các câu thành danh sách vector (Batch processing)
        Dùng cho việc Index dữ liệu nhanh hơn.
        """
        # LangChain HuggingFaceEmbeddings có sẵn hàm embed_documents
        return self.model.embed_documents(texts)

    def get_model(self):
        return self.model
    
    def encode(self, text: str) -> List[float]:
        # trả về List[float] thay vì object Embeddings của LangChain
        return self.model.embed_query(text)

# Singleton để dùng chung trong toàn bộ hệ thống
gemma_client = GemmaEmbeddingClient()
