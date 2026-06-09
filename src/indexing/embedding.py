import numpy as np
from typing import List, Optional
from src.core.config import EMBEDDING_MODEL_NAME, EMBEDDING_LOCAL_PATH, EMBEDDING_QUERY_INSTRUCTION, EMBEDDING_DIM


class BGEEmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME,
                 local_path: str = EMBEDDING_LOCAL_PATH,
                 query_instruction: str = EMBEDDING_QUERY_INSTRUCTION):
        self.model_name = model_name
        self.query_instruction = query_instruction
        self._model = None
        self._load_model(local_path)

    def _load_model(self, local_path: str):
        from sentence_transformers import SentenceTransformer
        model_path = local_path if local_path else self.model_name
        print(f"[BGE] 加载模型: {model_path}")
        self._model = SentenceTransformer(model_path)
        self._dim = self._model.get_embedding_dimension()
        print(f"[BGE] 模型加载完成, 向量维度: {self._dim}")

    @property
    def dim(self) -> int:
        return self._dim

    def encode_documents(self, texts: List[str], batch_size: int = 64,
                         show_progress: bool = True) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        if self.query_instruction:
            query = self.query_instruction + query
        emb = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(emb[0], dtype=np.float32)

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        if self.query_instruction:
            queries = [self.query_instruction + q for q in queries]
        embeddings = self._model.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embeddings, dtype=np.float32)


EmbeddingModel = BGEEmbeddingModel
