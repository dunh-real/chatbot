import sys
import asyncio
from pathlib import Path

# Thêm project root vào sys.path để import được folder services
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from services.ingestion.converter import document_converter

def print_separator(char="-", length=60):
    print(char * length)

async def test_single_file():
    # ===== CẤU HÌNH FILE TEST =====
    test_file_path = r"C:/AI Project/chatbot src/luat giao thong duong bo.pdf"
    
    file_path = Path(test_file_path)
    
    if not file_path.exists():
        print(f"❌ Không tìm thấy file tại: {test_file_path}")
        return

    print_separator("=")
    print(f"🚀 ĐANG TEST CONVERTER")
    print(f"📂 File nguồn: {file_path.name}")
    print_separator("=")

    try:
        # Gọi hàm convert (save_debug mặc định là True trong class DocumentConverter)
        print("⏳ Đang convert và làm sạch dữ liệu...")
        markdown_content = document_converter.convert(file_path, save_debug=True)
        
        # Đường dẫn file .md dự kiến
        debug_md_path = file_path.with_suffix('.md')

        print(f"✅ Convert thành công!")
        
        if debug_md_path.exists():
            print(f"📝 Đã tạo file MD tại: {debug_md_path}")
        else:
            print("⚠️ Cảnh báo: Không tìm thấy file .md được tạo ra.")

        # Xem trước kết quả
        print_separator()
        print("🔍 XEM TRƯỚC 1000 KÝ TỰ ĐẦU TIÊN:")
        print_separator()
        print(markdown_content[:1000])
        print_separator()
            
    except Exception as e:
        print(f"❌ LỖI: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_file())