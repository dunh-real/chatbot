from langchain_text_splitters import CharacterTextSplitter

with open("./file1.md", "r", encoding="utf-8") as f:
    markdown_document = f.read()



from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Bước 1: Định nghĩa các cấp độ Header muốn tách
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# Bước 2: Tách theo Header trước để giữ ngữ cảnh
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)

# Bước 3: Tinh chỉnh nhỏ hơn với RecursiveCharacterTextSplitter 
# (Dùng khi các đoạn dưới Header vẫn quá dài so với chunk_size)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""] # Ưu tiên ngắt ở đoạn văn, rồi mới đến dòng, rồi đến từ
)

# Thực hiện tách
final_chunks = text_splitter.split_documents(md_header_splits)

# # Kiểm tra kết quả
# print(f"Số lượng chunks: {len(final_chunks)}")
# for i in range(len(final_chunks)):
#     print(f"chunk {i+1}: \n{final_chunks[i].page_content}")
# # print(f"Nội dung chunk đầu tiên: \n{final_chunks[0].page_content}")
# print(f"Metadata (lưu Header): {final_chunks[0].metadata}")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# text_str = final_chunks[0].page_content
# embedd = model.encode(text_str)

embeddings = [model.encode(final_chunks[i].page_content) for i in range(len(final_chunks))]
# for i in range(len(embeddings)):
#     print(len(embeddings[i]))
#     print(type(embeddings[i]))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# initialize the client
client = QdrantClient(url = 'http://localhost:6333')

# create a collection
# client.create_collection(
#     collection_name = "test1_collection",
#     vectors_config = VectorParams(size = len(embeddings[0]), distance = Distance.DOT),
# )

# add vectors

# for i in range(len(final_chunks)):
#     operation_info = client.upsert(
#         collection_name = "test1_collection",
#         wait = True,
#         points = [
#             PointStruct(id = i, vector = [a for a in embeddings], payload = {"content": final_chunks[i].page_content}),
#         ],
#     )

operation_info = client.upsert(
    collection_name = "test1_collection",
    wait = True,
    points = [PointStruct(id = i, vector = embeddings[i], payload = {"content": final_chunks[i].page_content})  for i in range(len(final_chunks))],
)