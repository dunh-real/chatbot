import sys
import time
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import các services
ROOT_DIR = Path(__file__).resolve().parent.parent 
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Import service tìm kiếm
from services.retrieval.qdrant_service import vector_store_service

# --- CẤU HÌNH TEST ---
TENANT_ID = "test_integration_tenant_01" 

def print_separator(char="-", length=80):
    print(char * length)

def main():
    print_separator("=")
    print("🚀 TOOL TEST TÌM KIẾM (SEARCH ONLY MODE)")
    print(f"🏢 Tenant ID đang search: {TENANT_ID}")
    print("💡 Gõ 'exit' hoặc 'quit' để thoát chương trình.")
    print_separator("=")

    while True:
        # 1. Nhập câu hỏi từ bàn phím
        print("\n👇 Nhập câu hỏi của bạn:")
        query = input(">> ").strip()

        if query.lower() in ["exit", "quit"]:
            print("👋 Bye bye!")
            break
        
        if not query:
            continue

        # 2. Gọi hàm tìm kiếm
        start_time = time.time()
        try:
            results = vector_store_service.search_hybrid(
                query=query,
                tenant_id=TENANT_ID,
                k=5  # Lấy top 5 kết quả tốt nhất
            )
        except Exception as e:
            print(f"❌ Lỗi khi tìm kiếm: {e}")
            continue
        
        duration = time.time() - start_time

        # 3. Hiển thị kết quả
        print(f"\n🔎 Tìm thấy {len(results)} kết quả trong {duration:.4f}s:\n")
        
        if not results:
            print("❌ Không tìm thấy thông tin nào phù hợp.")
        else:
            for i, point in enumerate(results):
                payload = point.payload
                score = point.score
                
                # Lấy metadata
                file_name = payload.get('src_file', 'Unknown File')
                h1 = payload.get('h1', '')
                h2 = payload.get('h2', '')
                content = payload.get('content', '')
                
                # Tạo tiêu đề ngữ cảnh (Breadcrumb)
                context_path = f"{file_name}"
                if h1: context_path += f" > {h1}"
                if h2: context_path += f" > {h2}"

                print_separator("-")
                print(f"🏆 Rank {i+1} | Score: {score:.4f}")
                print(f"📂 Context: {context_path}")
                print(f"📖 Content: {content}...")
                print_separator("-")

if __name__ == "__main__":
    main()