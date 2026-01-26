import re
import os
from pathlib import Path

# Clean md file
def auto_fix_markdown_headers(text):
    lines = text.split('\n')
    fixed_lines = []
    
    # 1. Pattern loại bỏ Page markers (Ví dụ: ## Page 1, # Page 2, Page 3...)
    page_pattern = r'^#*\s*Page\s*\d+'
    
    # 2. Pattern nhận diện tiêu đề La Mã (I., II., III...)
    roman_pattern = r'^([IVXLCDM]+\.\s+[A-ZĐỨ]*)' 
    
    # 3. Pattern nhận diện tiêu đề số (1., 2., 3...)
    arabic_pattern = r'^(\d+\.\s+[A-ZĐỨa-zđứ]*)'

    for line in lines:
        stripped_line = line.strip()

        # BƯỚC 1: Xử lý dấu bôi đậm ** (Nếu có)
        # Loại bỏ tất cả dấu ** ở đầu và cuối dòng để lấy text thuần
        if stripped_line.startswith("**") and stripped_line.endswith("**"):
            stripped_line = stripped_line.strip("*").strip()
        
        # BƯỚC 2: Loại bỏ các dòng Page marker
        if re.match(page_pattern, stripped_line, re.IGNORECASE):
            continue  
            
        # BƯỚC 3: Giữ nguyên nếu đã là tiêu đề chuẩn hoặc dòng trống
        if not stripped_line:
            fixed_lines.append("")
            continue
            
        # BƯỚC 4: Xử lý tiêu đề La Mã -> Cấp 2 (##)
        if re.match(roman_pattern, stripped_line):
            fixed_lines.append(f"## {stripped_line}")
            
        # BƯỚC 5: Xử lý tiêu đề số -> Cấp 3 (###)
        elif re.match(arabic_pattern, stripped_line):
            fixed_lines.append(f"### {stripped_line}")
            
        # BƯỚC 6: Các nội dung văn bản khác giữ nguyên
        else:
            fixed_lines.append(line)
            
    # Gộp lại và xử lý khoảng trắng thừa nếu cần
    return '\n'.join(fixed_lines)

PATH_MD_DOCUMENT = "./md_file"
md_files = list(Path(PATH_MD_DOCUMENT).glob("*.md"))

PATH_MD_CLEANED = "./md_cleaned"

for md_file in md_files:
    raw_text = md_file.read_text(encoding="utf-8")

    result = auto_fix_markdown_headers(raw_text)
    
    file_name = os.path.basename(md_file)
    output_path = os.path.join(PATH_MD_CLEANED, file_name)

    with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
