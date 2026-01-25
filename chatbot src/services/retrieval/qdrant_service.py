import logging
from typing import List, Dict
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding # Thư viện tạo Sparse Vector (BM25) siêu nhẹ
from services.embedding.gemma_client import gemma_client
import uuid

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self):
        # 1. Kết nối Qdrant
        self.client = QdrantClient(url="http://localhost:6333") 
        self.collection_name = "chatbot_knowledge_for_businesses"
        self.vector_size = 768 # Gemma-300m
        
        # 2. Khởi tạo Model Sparse (BM25)
        logger.info("⏳ Đang tải model Sparse (BM25)...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("✅ Model Sparse đã sẵn sàng.")

        # 3. Đảm bảo Collection tồn tại
        self._ensure_collection()

    def _ensure_collection(self):
        """
        Tạo collection hỗ trợ cả Dense và Sparse vector, tối ưu cho Upload.
        """
        if not self.client.collection_exists(self.collection_name):
            logger.info(f"🛠️ Đang tạo collection Hybrid '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                
                # CẤU HÌNH DENSE VECTOR (Gemma)
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                        on_disk=True
                    )
                },
                
                # CẤU HÌNH SPARSE VECTOR (BM25)
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=True, # Tiết kiệm RAM
                        )
                    )
                },

                # CẤU HÌNH HNSW (Tắt index để upload nhanh)
                hnsw_config=models.HnswConfigDiff(m=0),
                
                shard_number=2 
            )

    def optimize_indexing(self):
        """
        Bật lại Indexing sau khi upload xong.
        """
        logger.info("⚙️ Đang bật lại Indexing m...")
        self.client.update_collection(
            collection_name=self.collection_name,
            hnsw_config=models.HnswConfigDiff(m=16)
        )
        logger.info("✅ Indexing đã được kích hoạt!")

    def add_chunks(self, chunks: List[Dict]):
            if not chunks:
                return

            points = []
            logger.info(f"💾 Đang chuẩn bị {len(chunks)} points để upload...")

            # Batch encode cho Dense Vector (Gemma) để nhanh hơn
            texts = [chunk['content'] for chunk in chunks]
            dense_vectors = gemma_client.embed_documents(texts) # Trả về list[list[float]]

            # Encode Sparse Vector (BM25)
            # bm25_client.encode_documents thường trả về generator hoặc list các object SparseEmbedding
            sparse_vectors = list(self.sparse_model.embed(texts))
            for i, chunk in enumerate(chunks):
                # 1. Xử lý Dense Vector
                # Đảm bảo nó là list float (nếu gemma_client trả về numpy thì phải .tolist())
                dense_vec = dense_vectors[i] 
                if hasattr(dense_vec, 'tolist'):
                    dense_vec = dense_vec.tolist()

                # 2. Xử lý Sparse Vector
                # Object SparseEmbedding có .indices và .values là numpy array -> Cần convert sang list
                raw_sparse = sparse_vectors[i]
                
                # Tạo object models.SparseVector chuẩn của Qdrant
                sparse_vec = models.SparseVector(
                    indices=raw_sparse.indices.tolist(), # Convert numpy -> list
                    values=raw_sparse.values.tolist()    # Convert numpy -> list
                )

                # 3. Tạo PointStruct
                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense_vec,    # Vector đặc (Semantic)
                        "sparse": sparse_vec   # Vector thưa (Keyword)
                    },
                    payload=chunk  # Metadata
                ))

            # Upload batch
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"✅ Đã upload thành công {len(points)} chunks vào Qdrant!")
            except Exception as e:
                logger.error(f"❌ Lỗi khi upsert vào Qdrant: {e}")
                raise e

    def search_hybrid(self, query: str, tenant_id: str, k: int = 10):
            # --- 1: TẠO VECTOR ---
            # 1.1. Tạo Dense Vector (Gemma)
            dense_vector = gemma_client.get_embedding(query)

            # 1.2. Tạo Sparse Vector (BM25)
            # FastEmbed trả về generator -> lấy item đầu tiên
            sparse_gen = list(self.sparse_model.embed(query))
            sparse_emb = sparse_gen[0]

            # Convert sang định dạng Qdrant
            qdrant_sparse_vector = models.SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist()
            )

            # --- 2: CẤU HÌNH RANK FUSION (RRF) ---
            # Lấy nhiều hơn k một chút (ví dụ k*2) để Fusion hiệu quả hơn
            prefetch_limit = k

            # 2.1. Prefetch bằng Sparse (Keyword Search)
            prefetch_sparse = models.Prefetch(
                query=qdrant_sparse_vector,
                using="sparse",
                limit=prefetch_limit,
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="tenant_id", 
                            match=models.MatchValue(value=tenant_id)
                        )
                    ]
                )
            )

            # 2.2. Prefetch bằng Dense (Semantic Search)
            prefetch_dense = models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=prefetch_limit,
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="tenant_id", 
                            match=models.MatchValue(value=tenant_id)
                        )
                    ]
                )
            )

            # --- 3: THỰC HIỆN TRUY VẤN FUSION ---
            results = self.client.query_points(
                collection_name=self.collection_name,
                
                # Đưa cả 2 chiến lược tìm kiếm vào danh sách prefetch
                prefetch=[prefetch_sparse, prefetch_dense],
                
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                
                limit=k, 

                with_payload=True,
            )

            return results.points

# Singleton
vector_store_service = VectorStoreService()