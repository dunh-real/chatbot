import sys
import asyncio
import io
import time
from pathlib import Path
from fastapi import UploadFile

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from services.ingestion.splitter import process_uploaded_file


def print_separator(char="=", length=80):
    print(char * length)


async def test_process_pdf():
    # ===== CONFIG =====
    pdf_path = r"C:/AI Project/chatbot src/luat giao thong duong bo.pdf"
    
    tenant_id = "bo_cong_an"

    file_path = Path(pdf_path)
    if not file_path.exists():
        print(f"❌ Không tìm thấy file: {pdf_path}")
        return

    print_separator()
    print(f"📂 FILE         : {file_path.name}")
    print(f"🏢 TENANT ID    : {tenant_id}")
    print(f"⚡ PIPELINE     : Hybrid (Markdown → Pure Semantic AI)")
    print(f"🧠 EMBEDDING    : google/embeddinggemma-300m")
    print(f"🚫 RECURSIVE    : OFF (Model tự quyết định độ dài)")
    print_separator()

    # ===== MOCK UploadFile =====
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    upload_file = UploadFile(
        filename=file_path.name,
        file=io.BytesIO(file_bytes),
        size=len(file_bytes),
    )

    try:
        print("⏳ Đang xử lý ingestion pipeline...")
        start = time.time()

        # CALL PIPELINE
        chunks = await process_uploaded_file(upload_file, tenant_id)

        elapsed = time.time() - start

        print(f"\n✅ HOÀN THÀNH sau {elapsed:.2f}s")
        print(f"📦 Tổng chunks : {len(chunks)}")
        print_separator()

        if not chunks:
            print("⚠️ Không tạo được chunk nào")
            return

        # ===== SAMPLE DEBUG =====
        # Lấy mẫu ở giữa tài liệu để tránh Mở đầu/Kết luận
        sample_n = min(3, len(chunks))
        mid = len(chunks) // 2
        samples = chunks[mid : mid + sample_n]

        print(f"🔍 SAMPLE {sample_n} CHUNKS (giữa tài liệu)\n")

        for i, chunk in enumerate(samples, 1):
            content = chunk["content"]
            meta = chunk["metadata"]

            print(f"🔹 CHUNK #{i}")
            print(f"🆔 ID         : {chunk['chunk_id']}")
            print(f"🛠️ METHOD     : {meta.get('method')}")
            print(f"📏 LENGTH     : {meta.get('length')} chars")
            
            # Kiểm tra xem Header Context có được thêm vào không
            if "**Ngữ cảnh:" in content:
                print("✅ CONTEXT    : Header injected")
            else:
                print("ℹ️ CONTEXT    : Flat text (Không có H1/H2 cha)")

            print("\n📄 CONTENT:")
            print("-" * 50)
            print(content)
            print("-" * 50)
            print()

        # ===== STATS =====
        lengths = [len(c["content"]) for c in chunks]
        avg_len = int(sum(lengths) / len(lengths))
        max_len = max(lengths)

        print_separator("-")
        print("📊 THỐNG KÊ CHI TIẾT")
        print(f"• Tổng chunks      : {len(chunks)}")
        print(f"• Ngắn nhất        : {min(lengths)} chars")
        print(f"• Trung bình       : {avg_len} chars")
        print(f"• Dài nhất         : {max_len} chars")

        # Cảnh báo mềm về Token Window
        # Giả sử 1 token ~ 4 chars (Tiếng Việt có thể ít hơn, tầm 3 chars)
        est_tokens = max_len / 3.5 
        print(f"• Max Tokens (est) : ~{int(est_tokens)} tokens")

        print("-" * 30)

    except Exception as e:
        print("❌ ERROR:", e)
        import traceback
        traceback.print_exc()

    finally:
        await upload_file.close()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(test_process_pdf())