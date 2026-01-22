from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer
import os
import torch

# Model Name
DENSE_MODEL_NAME = "BAAI/bge-m3"
SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"

# Save Model
MODEL_CACHE_FOLDER = os.path.join(os.path.dirname(__file__), "models_cache")
os.makedirs(MODEL_CACHE_FOLDER, exist_ok=True)

# Make embedding dense-vector
class LocalDenseEmbedding:
    def __init__(self, model_name=DENSE_MODEL_NAME, cache_folder=MODEL_CACHE_FOLDER):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = SentenceTransformer(
            model_name, 
            device=device,
            cache_folder=cache_folder
        )

    def embed(self, texts: list[str]): 
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        return embeddings.tolist()

# Make embedding sparse-vector
class LocalSparseEmbedding:
    def __init__(self, model_name=SPARSE_MODEL_NAME, cache_folder=MODEL_CACHE_FOLDER):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_folder)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_folder)
        self.model.to(self.device) 
        self.model.eval() 

    def embed(self, texts: list[str]):
        results = []
        
        with torch.no_grad(): 
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                
                outputs = self.model(**inputs)
                logits = outputs.logits

                relu_logits = torch.relu(logits)
                
                log_logits = torch.log(1 + relu_logits)
                
                masked_logits = log_logits * inputs.attention_mask.unsqueeze(-1)
                
                sparse_vec_tensor, _ = torch.max(masked_logits, dim=1)
                sparse_vec_tensor = sparse_vec_tensor.squeeze() 

                indices = sparse_vec_tensor.nonzero().squeeze().cpu().tolist()
  
                values = sparse_vec_tensor[indices].cpu().tolist()
                
                if isinstance(indices, int): indices = [indices]
                if isinstance(values, float): values = [values]

                results.append({"indices": indices, "values": values})
                
        return results