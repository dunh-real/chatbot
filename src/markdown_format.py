import re
from typing import List, Tuple

class MarkdownReformat:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def load_md_file(self, markdown_path):
        with open(markdown_path, "r", encoding = "latin-1") as f:
            return f.read()
    
    def normalize_heading(self, content: str) -> str:
        lines = content.split('\n')
        normalized = []
        prev_was_heading = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # skip empty lines after headings (we'll add them back consistently)
            if not stripped and prev_was_heading:
                continue

            # detect and normalize headings
            if stripped.startswith('#'):
                # add blank line before heading (except first line)
                if normalized and normalized[-1].strip():
                    normalized.append('')
                
                # normalize heading format (remove extra spaces)
                heading_match = re.match(r'^(#+)\s(.+)$', stripped)
                if heading_match:
                    level, text = heading_match.groups()
                    normalized.append(f"{level} {text.strip()}")
                else:
                    normalized.append(stripped)
                
                # add blank line after heading
                normalized.append('')
                prev_was_heading = True
            else:
                normalized.append(line.rstrip())
                prev_was_heading = False
        
        return '\n'.join(normalized)
    
    def fix_list_formatting(self, content: str) -> str:
        # fix bullet points and numbered lists formatting
        lines = content.split('\n')
        fixed = []
        in_list = False

        for line in lines:
            stripped = line.strip()

            # detect list items
            is_bullet = re.match(r'^[-*+]\s+.+', stripped)
            is_numbered = re.match(r'^\d+\.\s+.+', stripped)

            if is_bullet or is_numbered:
                if not in_list and fixed and fixed[-1].strip():
                    fixed.append('') # add space before list
                in_list = True
                # normalize bullet points to use '-'
                if is_bullet:
                    text = re.sub(r'^[-*+]\s+', '- ', stripped)
                    fixed.append(text)
                else:
                    fixed.append(stripped)
            else:
                if in_list and stripped:
                    in_list = False
                    if fixed and fixed[-1].strip():
                        fixed.append('') # add space after list
                fixed.append(line.rstrip())
        
        return '\n'.join(fixed)
    
    def remove_excessive_whitespace(self, content: str) -> str:
        # remove excessive blank lines while maintaining structure
        # replace 3+ consecutive newlines with 2
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip() + '\n'
    
    def fix_inline_formatting(self, content: str) -> str:
        # fix inline formatting issues (bold, italic, code)
        content = re.sub(r'\*\*\s+', '**', content)
        content = re.sub(r'\s+\*\*', '**', content)

        content = re.sub(r'(?<!\*)\*\s+', '*', content)
        content = re.sub(r'\s+\*(?!\*)', '*', content)

        content = re.sub(r'`\s+', '`', content)
        content = re.sub(r'\s+`', '`', content)

        return content
    
    def organize_content(self, content: str) -> str:
        # main organization pipeline
        # apply all formatting fixes
        content = self.normalize_heading(content)
        content = self.fix_list_formatting(content)
        content = self.fix_inline_formatting(content)
        content = self.remove_excessive_whitespace(content)

        return content
    
    def add_metadata(self, content: str) -> str:
        # add metadata header for embedding preparation
        metadata = """---
        # Document reformatted for embedding preparation
        # Organized structure for optimal chunking
        ---"""

        return metadata + '\n' + content
    
    def process(self, add_metadata: bool = True) -> str:
        # process the markdown file and return reformatted content
        content = self.load_md_file(self.input_file)
        content = self.organize_content(content)

        if add_metadata:
            content = self.add_metadata(content)
        
        return content
    
    def save(self, content: str = None):
        # save the reformatted content to output file
        if content is None:
            content = self.process()
        
        with open(self.output_file, 'w', encoding = 'latin-1') as f:
            f.write(content)
        return self.output_file


new_format = MarkdownReformat('../file1.md', './file1.md')
new_format.process()
new_format.save()