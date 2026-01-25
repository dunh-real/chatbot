import sys
import asyncio
import io
import time
from pathlib import Path
from fastapi import UploadFile

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent 
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# 🔥 IMPORT QUAN TRỌNG
from services.ingestion.splitter import process_uploaded_file 
from services.retrieval.qdrant_service import vector_store_service

def print_separator(char="=", length=80):
    print(char * length)

async def run_full_integration_test():
    # ===== CONFIG =====
    pdf_path = r"C:/AI Project/chatbot src/luat giao thong duong bo.pdf"
    
    tenant_id = "test_integration_tenant_01" 
    query_test = "mức phạt nồng độ cồn là bao nhiêu" 

    file_path = Path(pdf_path)
    if not file_path.exists():
        print(f"❌ Không tìm thấy file: {pdf_path}")
        return

    print_separator()
    print("🚀 BẮT ĐẦU TEST TOÀN TRÌNH (INGESTION -> STORE -> SEARCH)")
    print_separator()
    print(f"📂 FILE         : {file_path.name}")
    print(f"🏢 TENANT ID    : {tenant_id}")
    print(f"🧠 MODEL        : Google Gemma + BM25 (Hybrid)")
    print_separator()

    # ===== GIAI ĐOẠN 1: MOCK UPLOAD & INGESTION =====
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    upload_file = UploadFile(
        filename=file_path.name,
        file=io.BytesIO(file_bytes),
        size=len(file_bytes),
    )

    chunks = []
    try:
        print("\n⏳ [PHASE 1] Đang xử lý Ingestion Pipeline...")
        print("   (Convert PDF -> Markdown -> Split)")
        
        start = time.time()

        # 1. Gọi hàm xử lý để lấy Chunks
        result = await process_uploaded_file(upload_file, tenant_id)
        
        if isinstance(result, list):
            chunks = result
            print(f"✅ Đã cắt thành công: {len(chunks)} chunks")

            # Vì process_uploaded_file chỉ trả về list, ta phải gọi store để lưu
            if len(chunks) > 0:
                print(f"💾 Đang gọi add_chunks để lưu {len(chunks)} chunks vào Qdrant...")
                vector_store_service.add_chunks(chunks)
                print("✅ Đã lưu chunks vào Database thành công!")
            else:
                print("⚠️ File rỗng hoặc không tạo được chunk nào.")
        
        elif isinstance(result, dict) and "chunks_count" in result:
            print(f"✅ Ingestion báo thành công: {result['message']}")
            # Nếu hàm trả về dict nghĩa là nó đã tự lưu bên trong
            # Nhưng để chắc chắn cho việc test search, ta cần đảm bảo có dữ liệu
            chunks = [1] * result["chunks_count"] 
        
        elapsed = time.time() - start
        print(f"✅ [PHASE 1] HOÀN THÀNH sau {elapsed:.2f}s")
        
    except Exception as e:
        print("❌ ERROR tại Phase 1:", e)
        import traceback
        traceback.print_exc()
        return
    finally:
        await upload_file.close()

    # ===== GIAI ĐOẠN 2: SEARCH VERIFICATION =====
    print("\n")
    print_separator("-")
    print(f"🔎 [PHASE 2] KIỂM TRA TÌM KIẾM TRÊN QDRANT")
    print(f"❓ Query thử nghiệm: '{query_test}'")
    print("⏳ Đang đợi 2s để Qdrant index xong...")
    time.sleep(2) 

    try:
        # Gọi hàm Search Hybrid
        results = vector_store_service.search_hybrid(
            query=query_test,
            tenant_id=tenant_id,
            k=5
        )

        if not results:
            print("❌ KẾT QUẢ: Không tìm thấy gì! (Có thể lỗi embedding hoặc chưa lưu được)")
        else:
            print(f"🎉 KẾT QUẢ: Tìm thấy {len(results)} chunks phù hợp!\n")
            
            for i, point in enumerate(results):
                payload = point.payload
                score = point.score
                
                print(f"🏆 Rank {i+1} | Score: {score:.4f}")
                print(f"   📂 File: {payload.get('src_file')}")
                print(f"   🏷️  H1: {payload.get('h1', 'N/A')}")
                print(f"   🏷️  H2: {payload.get('h2', 'N/A')}")
                
                # Cắt ngắn content để hiển thị
                content_preview = payload.get('content', '')[:200].replace('\n', ' ')
                print(f"   📖 Content: {content_preview}...")
                print("-" * 50)

    except Exception as e:
        print("❌ ERROR tại Phase 2 (Search):", e)
        import traceback
        traceback.print_exc()

    print_separator()
    print("🏁 TEST KẾT THÚC")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_full_integration_test())