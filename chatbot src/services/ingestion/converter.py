import re
import logging
from pathlib import Path
from fastapi import HTTPException
from markitdown import MarkItDown
import pymupdf4llm

logger = logging.getLogger(__name__)

class DocumentConverter:
    def __init__(self):
        self.office_converter = MarkItDown()

    def convert(self, file_path: Path, save_debug: bool = True) -> str:
        """
        Chuyển đổi file sang Markdown và lưu bản sao .md để debug.
        """
        ext = file_path.suffix.lower()
        try:
            # 1. chuyển đổi
            if ext == ".pdf":
                text = pymupdf4llm.to_markdown(str(file_path))
            else:
                text = self.office_converter.convert(str(file_path)).text_content
            
            # 2. Làm sạch văn bản
            cleaned_text = self._cleanup_and_normalize(text)
            
            # 3. LƯU FILE .MD (DEBUG)
            if save_debug:
                self._save_md_debug(file_path, cleaned_text)
                
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            raise HTTPException(500, f"Lỗi convert file: {str(e)}")

    def _save_md_debug(self, original_file_path: Path, content: str):
        """
        Lưu nội dung đã convert vào file .md cùng vị trí với file gốc.
        """
        try:
            # Tạo tên file: "ten_file.pdf" -> "ten_file.md"
            debug_path = original_file_path.with_suffix('.md')
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"✅ Đã lưu file debug Markdown tại: {debug_path}")
        except Exception as e:
            logger.warning(f"⚠️ Không thể lưu file debug .md: {e}")

    def _cleanup_and_normalize(self, text: str) -> str:
        """
        Làm sạch văn bản thô.
        """
        # Xử lý Header dính
        text = re.sub(r'([\.\;\n])\s*(\*\*[^*\n]{3,100}\*\*)', r'\1\n## \2', text)

        lines = text.split('\n')
        cleaned_lines = []
        universal_numbering = re.compile(r'^(\d+(\.\d+)*|[IVX]+|[A-Z]|[a-z])[\.\)\-]\s+')

        for line in lines:
            stripped = line.strip()
            if not stripped: continue
            if re.match(r'^[-_\s]*\d+[-_\s]*$', stripped): continue 
            if re.match(r'^[_=*-]{3,}$', stripped): continue 
            
            if len(stripped) < 200:
                if stripped.startswith('##'):
                    cleaned_lines.append(stripped.replace('**', ''))
                    continue
                if stripped.isupper() and len(stripped) > 4:
                    cleaned_lines.append(f"## {stripped}")
                    continue
                if stripped.startswith('**') and stripped.endswith('**'):
                    cleaned_lines.append(f"### {stripped.replace('**', '')}")
                    continue
                if universal_numbering.match(stripped):
                    cleaned_lines.append(f"### {stripped}")
                    continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

document_converter = DocumentConverter()