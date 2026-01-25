import sys
import time
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage

# --- CẤU HÌNH ĐƯỜNG DẪN IMPORT ---
ROOT_DIR = Path(__file__).resolve().parent.parent 
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Import các service
from services.retrieval.qdrant_service import vector_store_service
from services.generation.llm_service import get_llm_service

# Tenant ID giả định (phải khớp với ID lúc upload file)
TENANT_ID = "test_integration_tenant_01" 

def print_separator(char="-", length=80):
    print(char * length)

def main():
    # 1. Khởi tạo LLM
    try:
        llm = get_llm_service()
        print("✅ Đã kết nối thành công với LLM Service (Groq).")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo LLM: {e}")
        return

    print_separator("=")
    print("🤖 TEST RAG FULL FLOW (QDRANT + GROQ LLM)")
    print(f"🏢 Tenant ID: {TENANT_ID}")
    print_separator("=")

    while True:
        # 2. Nhập câu hỏi
        print("\n👇 Nhập câu hỏi (gõ 'exit' để thoát):")
        query = input(">> ").strip()

        if query.lower() in ["exit", "quit"]:
            break
        if not query:
            continue

        start_time = time.time()

        # 3. Bước Retrieval (Tìm kiếm từ Qdrant)
        print("🔍 Đang tìm kiếm thông tin liên quan...")
        try:
            search_results = vector_store_service.search_hybrid(
                query=query,
                tenant_id=TENANT_ID,
                k=5 
            )
        except Exception as e:
            print(f"❌ Lỗi khi tìm kiếm Qdrant: {e}")
            continue
        
        if not search_results:
            print("⚠️ Không tìm thấy tài liệu nào phù hợp. LLM sẽ trả lời bằng kiến thức nền.")
            context_text = "Không có thông tin ngữ cảnh cụ thể."
        else:
            # Gộp nội dung các chunk tìm được thành chuỗi context
            context_chunks = [f"--- Nguồn: {p.payload.get('src_file', 'Unknown')} ---\n{p.payload.get('content')}" for p in search_results]
            context_text = "\n\n".join(context_chunks)

        # 4. Bước Generation (Sinh câu trả lời với LangChain)
        print("🧠 Đang suy nghĩ và tổng hợp câu trả lời...")
        
        # Tạo Prompt System (Chứa ngữ cảnh)
        system_prompt = f"""Bạn là trợ lý AI chuyên về Luật.
        Hãy trả lời câu hỏi dựa trên thông tin sau:
        
        {context_text}
        
        Nếu thông tin không có trong ngữ cảnh, hãy nói rõ là không tìm thấy trong tài liệu được cung cấp.
        Trả lời ngắn gọn, chuyên nghiệp bằng Tiếng Việt."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        try:
            # Gọi LLM
            response = llm.invoke(messages)
            answer = response.content
        except Exception as e:
            print(f"❌ Lỗi khi gọi LLM: {e}")
            continue

        total_time = time.time() - start_time

        # 5. Hiển thị kết quả
        print("\n" + "="*30 + " 💡 CÂU TRẢ LỜI " + "="*30)
        print(answer)
        print("="*80)
        print(f"⏱️ Tổng thời gian xử lý: {total_time:.2f}s")

if __name__ == "__main__":
    main()