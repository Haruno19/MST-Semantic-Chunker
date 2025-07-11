from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

class EmbeddingFunction:
    def __init__(self, embedder="Qwen/Qwen3-Embedding-0.6B"):
        self.embedder = SentenceTransformer(embedder)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

    def embed(self, chunks:list):
        return self.embedder.encode(chunks).tolist()

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))