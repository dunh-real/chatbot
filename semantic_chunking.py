import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# configuration
file_path = './data_bienban.md'
embedding_model = 'all-MiniLM-L6-v2'

def load_markdown_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} was not found.")
    
    with open(path, 'r', encoding = 'utf-8') as f:
        return f.read()
    
def main():
    print(f"-- Loading file: {file_path} --")
    try:
        markdown_text = load_markdown_file(file_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("-- Initializing embedding model --")
    embeddings = HuggingFaceEmbeddings(model_name = embedding_model)

    print("-- Configuring semantic chunker --")
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type = "percentile"
    )

    print("-- Chunking text --")
    docs = text_splitter.create_documents([markdown_text])
    print(f"\nSuccess! Created {len(docs)} semantic chunks.\n")

    # output results
    print("-- Preview of chunks --")
    for i, doc in enumerate(docs):
        content = doc.page_content.replace('\n', ' ')
        print(f"\n[Chunk {i+1}] (Length: {len(content)} chars)")
        print(f"Content: {content}")
        print('-'*70)

if __name__ == "__main__":
    main()