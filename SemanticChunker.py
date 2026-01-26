import torch
import os
import nltk
import re
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

MODEL_NAME = "BAAI/bge-m3"
cache_folder = os.path.join(os.path.dirname(__file__), "models_cache")
os.makedirs(cache_folder, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
        
# Load Model
model = SentenceTransformer(
    MODEL_NAME, 
    device=device,
    cache_folder=cache_folder
)

# Compare Context
def should_break_here(context_before, context_after, threshold = 0.5):
    if not context_before.strip() or not context_after.strip():
        return False

    embed_before = model.encode(context_before, convert_to_tensor=True)
    embed_after = model.encode(context_after, convert_to_tensor=True)

    # Cosine Similarity
    score = util.cos_sim(embed_before, embed_after).item()

    return score < threshold

# Custom Sentences
def custom_sent_tokenize(text):
    raw_sentences = sent_tokenize(text)
    
    merged_sentences = []
    buffer = ""
    
    re_list_marker = r'^[\s\*\#]*(\d+(\.\d+)*|[a-zA-Z])[\.\)][\s\*\#]*$'
    
    re_legal_marker = r'^[\s\*\#]*(Điều|Khoản|Chương|Mục|Phần|Tiểu mục)\s+\d+.*[\.\:]?[\s\*\#]*$'
    
    re_admin_marker = r'^[\s\*\#]*(QUYẾT ĐỊNH|THÔNG BÁO|CÔNG ĐIỆN|YÊU CẦU|KẾT LUẬN|CHỈ THỊ).*[:\.]?[\s\*\#]*$'

    re_orphan_number = r'^[\s\*\#]*\d+[\s\*\#]*$'

    for sent in raw_sentences:
        s_stripped = sent.strip()
        if not s_stripped: continue
        
        is_prefix = False
        
        if re.match(re_list_marker, s_stripped) or \
           re.match(re_legal_marker, s_stripped, re.IGNORECASE) or \
           re.match(re_admin_marker, s_stripped, re.IGNORECASE) or \
           re.match(re_orphan_number, s_stripped):
            
            if len(s_stripped.split()) < 15:
                is_prefix = True

        if s_stripped.endswith(':'):
             is_prefix = True
        
        if is_prefix:
            buffer += s_stripped + " "
        else:
            full_sent = buffer + s_stripped
            merged_sentences.append(full_sent.strip())
            buffer = "" 
            
    if buffer:
        merged_sentences.append(buffer.strip())
        
    return merged_sentences

def context_aware_chunking(text):
    sentences = custom_sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for i, sentence in enumerate(sentences):
        context_before = sentences[i-1] if i > 0 else ""
        context_after = sentence
        if i > 0 and should_break_here(context_before, context_after, threshold=0.7):
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

file_path = "./md_cleaned/document_6_output.md"

with open(file_path, "r", encoding="utf-8") as f:
    docs = f.read()

chunks = context_aware_chunking(docs)

idx = 0
for i in chunks:
    idx += 1
    print("=="*30)
    print(f"[CHUNK {idx}]:", i)
    print("=="*30)
