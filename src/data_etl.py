"""
INPUT DATA PREPROCESSING:

Input: markdown file from OCR model
Output: tuples include: chunks and embedding vectors

ETL Pipeline:
Step 1: Reformat the input markdown file to be a well organized markdown document.
Step 2: Chunking
Step 3: Convert chunks into embedding vectors
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Data_ETL:
    def __init__(self, input_path):
        self.input_path = input_path
        self.markdown_file = self.load_markdown_file(input_path)

    def load_markdown_file(self, markdown_path):
        with open(markdown_path, 'r', encoding = 'utf-8') as f:
            return f.read()

    
    def chunking(self, content, threshold):
        # 1. Split text into sentences (simple approach, can be improved with a dedicated NLP library)
        sentences = content.split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip]
        # 2. Load a local model
        # This downloads the model to your local machine the first time it runs
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # 3. Generate embeddings for sentences
        embeddings = model.encode(sentences)

        # 4. Calculate cosine similarity between adjacent sentences
        # We compare sentence[i] with sentence[i+1]
        similarities = []
        for i in range(len(embeddings) - 1):
            # Reshape for sklearn's cosine_similarity function
            s1 = embeddings[i].reshape(1, -1)
            s2 = embeddings[i+1].reshape(1, -1)
            similarity = cosine_similarity(s1, s2)[0][0]
            similarities.append(similarity)

        # 5. Determine cut points based on a threshold
        # Lower similarity indicates a topic shift
        cut_points = [i for i, sim in enumerate(similarities) if sim < threshold]

        # 6. Chunk the text based on cut points
        chunks = []
        current_chunk = []
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            if i in cut_points:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        # Add the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    
    def make_embedding_vectors(self, chunks):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = [model.encode(chunks[i].page_content) for i in range(len(chunks))]
        return embeddings

