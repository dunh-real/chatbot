import uuid
import os
from pathlib import Path
from ModelEmbed import LocalDenseEmbedding, LocalSparseEmbedding
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

# Setup DB
QDRANT_URL = "http://localhost:6333" 
COLLECTION_NAME = "enterprise_documents" 
DENSE_VECTOR_NAME = "dense-vector" 
SPARSE_VECTOR_NAME = "sparse-vector" 
DENSE_DIMENSION = 1024 

# Setup markdown file path
PATH_MD_DOCUMENT = "./md_file"
md_files = list(Path(PATH_MD_DOCUMENT).glob("*.md"))

def process_markdown_file(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            markdown_document = f.read()
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại {file_path}")
        return None, None, None

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])

    final_chunks = text_splitter.split_documents(md_header_splits)

    texts_to_embed = [chunk.page_content for chunk in final_chunks]

    # Dense vector
    dense_embedder = LocalDenseEmbedding()
    dense_vectors = dense_embedder.embed(texts_to_embed)
    
    # Sparse vector
    sparse_embedder = LocalSparseEmbedding()
    sparse_vectors = sparse_embedder.embed(texts_to_embed)

    return final_chunks, dense_vectors, sparse_vectors

def init_qdrant_collection(client: QdrantClient):
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        # Create collection
        client.create_collection(
            collection_name=COLLECTION_NAME,

            # Create dense - vector
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=DENSE_DIMENSION,
                    distance=Distance.COSINE
                )
            },

            # Create sparse - vector
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False, 
                    )
                )
            }
        )
        
        # Create payload (meta data)
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="tenant_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
    else:
        print("Collection đã tồn tại. Bỏ qua bước tạo.")

def upload_data_to_qdrant(chunks, dense_vecs, sparse_vecs, tenant_id="demo_tenant"):
    client = QdrantClient(url=QDRANT_URL)
    
    init_qdrant_collection(client)
    
    points = []

    for i, (chunk, dense, sparse) in enumerate(zip(chunks, dense_vecs, sparse_vecs)):
        
        payload = chunk.metadata.copy() 
        payload["page_content"] = chunk.page_content 
        payload["tenant_id"] = tenant_id 
        payload["file_name"] = os.path.basename(md_file)

        # Tạo PointStruct 
        point = models.PointStruct(
            id=str(uuid.uuid4()), # Tạo ID ngẫu nhiên duy nhất cho mỗi point
            vector={
                DENSE_VECTOR_NAME: dense,
                SPARSE_VECTOR_NAME: sparse
            },
            payload=payload
        )
        points.append(point)
        
    operation_info = client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True, 
        points=points
    )

if __name__ == "__main__":
    for md_file in md_files:

        client = QdrantClient(url=QDRANT_URL)

        CURRENT_TENANT_ID = "VGP"

        chunks, dense_vecs, sparse_vecs = process_markdown_file(md_file)

        # Push data to Qdrant
        if chunks and dense_vecs and sparse_vecs:
            upload_data_to_qdrant(
                chunks=chunks,
                dense_vecs=dense_vecs,
                sparse_vecs=sparse_vecs,
                tenant_id=CURRENT_TENANT_ID
            )
        else:
            print("\nCó lỗi trong quá trình xử lý file. Không thể upload.")
        
        