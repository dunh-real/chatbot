import ollama
from qdrant_client import QdrantClient, models
from ModelEmbed import LocalDenseEmbedding, LocalSparseEmbedding

client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "enterprise_documents"

def query_rag(user_query):
    # PROCESSING QUERY 
    dense_embedding = LocalDenseEmbedding() 
    dense_vector = dense_embedding.get_dense_vector(user_query)

    sparse_embedding = LocalSparseEmbedding()
    sparse_vector = sparse_embedding.get_sparse_vector(user_query)

    search_result = client.query_points(
        collection_name = COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query = dense_vector,
                using = "dense-vector",
                limit = 5
            ),
            models.Prefetch(
                query = sparse_vector,
                using = "sparse-vector",
                limit = 5
            )
        ],
        query = models.FusionQuery(fusion=models.Fusion.RRF),
    )

    context_list = []
    for hit in search_result.points:
        content = hit.payload.get("page_content", "")
        score = hit.score
        context_list.append(f"[Score: {score:.4f}] {content}")

    context_text = "\n\n".join(context_list)

    # PROMPT AUGMENTATION
    system_prompt = (
        """ 
        Bạn là một chuyên gia quản lý tri thức doanh nghiệp. 
        Nhiệm vụ của bạn là hỗ trợ người dùng bằng cách cung cấp thông tin chính xác, khách quan dựa trên tài liệu được cung cấp.
        QUY TẮC PHẢN HỒI:
            1. TRUNG THỰC TUYỆT ĐỐI: Chỉ sử dụng thông tin trong phần "Ngữ cảnh" (Context). Không sử dụng kiến thức bên ngoài hoặc tự suy diễn.
            2. PHÂN TÍCH LOGIC: Nếu ngữ cảnh có nhiều thông tin, hãy tổng hợp và trình bày theo các ý chính (sử dụng bullet points).
            3. XỬ LÝ KHI THIẾU THÔNG TIN: Nếu câu trả lời không có trong ngữ cảnh hoặc ngữ cảnh bị mờ nhạt, bạn phải phản hồi chính xác như sau: 
            "Tôi xin lỗi, thông tin bạn tìm kiếm hiện không có trong dữ liệu nội bộ của hệ thống. Vui lòng cung cấp thêm chi tiết hoặc liên hệ quản trị viên."
            4. PHONG CÁCH: Lịch sự, chuyên nghiệp, ngôn ngữ gãy gọn. Tránh các từ ngữ cảm thán thừa thãi.
            5. TRÍCH DẪN (Nếu có): Nếu trong ngữ cảnh có tên tài liệu hoặc số trang, hãy chỉ rõ nguồn khi trả lời.
        """
    )
    
    rag_prompt = f"""
    Ngữ cảnh (Context):
    {context_text}

    Câu hỏi của người dùng:
    {user_query}

    Hệ thống đưa ra câu trả lời:
    """

    print("--- THINKING ---")

    response = ollama.chat(model='llama3.2:1b', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': rag_prompt},
    ])

    return response['message']['content'], context_list

user_input = "Hôm nay là thứ mấy ?"
answer, source_docs = query_rag(user_input)

print("\n=== CHATBOT RESPONSE ===")
print(answer)
print("\n=== CONTEXT LIST ===")
for doc in source_docs:
    print(f"- {doc[:300]}...")