import os
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
from src.core.config import VECTOR_STORE_DIR, VECTOR_DEDUP_THRESHOLD, VECTOR_QUALITY_FILTER_RATIO


class FAISSVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.id_to_metadata: Dict[int, Dict] = {}
        self.current_id = 0

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vecs / norms

    def add_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        embeddings = self._normalize(embeddings).astype(np.float32)

        self.index.add(embeddings)

        for meta in metadata_list:
            self.id_to_metadata[self.current_id] = meta
            self.current_id += 1

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float, Dict]]:
        if self.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = self._normalize(query_embedding).astype(np.float32)

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, actual_k)

        results = []
        for i in range(actual_k):
            idx = int(indices[0][i])
            score = float(scores[0][i])
            if score > 0 and idx in self.id_to_metadata:
                results.append((idx, score, self.id_to_metadata[idx]))
        return results

    def get_total_count(self) -> int:
        return self.index.ntotal

    def dedup_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict],
                         threshold: float = VECTOR_DEDUP_THRESHOLD) -> Tuple[np.ndarray, List[Dict]]:
        if len(embeddings) <= 1:
            return embeddings, metadata_list

        normed = self._normalize(embeddings.copy()).astype(np.float32)
        n = len(normed)

        sim_matrix = normed @ normed.T
        np.fill_diagonal(sim_matrix, 0)

        keep_indices = []
        removed = set()

        order = np.argsort(-sim_matrix.max(axis=1))
        for i in order:
            if i in removed:
                continue
            keep_indices.append(i)
            duplicates = np.where(sim_matrix[i] > threshold)[0]
            for j in duplicates:
                if j != i and j not in removed:
                    removed.add(j)

        keep_indices = sorted(keep_indices)

        if removed:
            print(f"[VECTOR-DEDUP] 去除 {len(removed)} 个冗余向量 (阈值={threshold}, {n} -> {len(keep_indices)})")

        kept_embeddings = embeddings[keep_indices]
        kept_metadata = [metadata_list[i] for i in keep_indices]
        return kept_embeddings, kept_metadata

    def quality_filter_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict],
                                  filter_ratio: float = VECTOR_QUALITY_FILTER_RATIO) -> Tuple[np.ndarray, List[Dict]]:
        if len(embeddings) <= 10 or filter_ratio <= 0:
            return embeddings, metadata_list

        n = len(embeddings)
        normed = self._normalize(embeddings.copy()).astype(np.float32)

        quality_scores = np.zeros(n, dtype=np.float32)

        uniqueness_scores = self._compute_uniqueness(normed)
        max_u = uniqueness_scores.max() if uniqueness_scores.max() > 0 else 1.0
        uniqueness_scores = uniqueness_scores / max_u

        density_scores = self._compute_density(normed)
        max_d = density_scores.max() if density_scores.max() > 0 else 1.0
        density_scores = density_scores / max_d

        text_quality_scores = self._compute_text_quality(metadata_list)
        max_t = text_quality_scores.max() if text_quality_scores.max() > 0 else 1.0
        text_quality_scores = text_quality_scores / max_t

        quality_scores = 0.4 * uniqueness_scores + 0.3 * density_scores + 0.3 * text_quality_scores

        remove_count = max(1, int(n * filter_ratio))
        sorted_indices = np.argsort(quality_scores)
        remove_set = set(sorted_indices[:remove_count].tolist())

        keep_indices = [i for i in range(n) if i not in remove_set]

        if len(keep_indices) < n:
            print(f"[VECTOR-QUALITY] 质量筛选移除 {len(remove_set)} 个低质量向量 "
                  f"(比例={filter_ratio}, {n} -> {len(keep_indices)})")

        kept_embeddings = embeddings[keep_indices]
        kept_metadata = [metadata_list[i] for i in keep_indices]
        return kept_embeddings, kept_metadata

    def _compute_uniqueness(self, normed: np.ndarray) -> np.ndarray:
        n = len(normed)
        uniqueness = np.ones(n, dtype=np.float32)

        if n <= 2:
            return uniqueness

        k = min(20, n - 1)
        sim_matrix = normed @ normed.T
        np.fill_diagonal(sim_matrix, 0)

        for i in range(n):
            row = sim_matrix[i]
            top_k_idx = np.argpartition(-row, min(k-1, n-1))[:k]
            neighbor_sims = row[top_k_idx]
            avg_sim = neighbor_sims.mean()
            max_sim = neighbor_sims.max()
            uniqueness[i] = 1.0 - 0.5 * avg_sim - 0.5 * max_sim

        return uniqueness

    def _compute_density(self, normed: np.ndarray) -> np.ndarray:
        n = len(normed)
        density = np.zeros(n, dtype=np.float32)

        centroid = normed.mean(axis=0, keepdims=True)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        sims_to_centroid = (normed @ centroid.T).flatten()
        density = sims_to_centroid

        return density

    def _compute_text_quality(self, metadata_list: List[Dict]) -> np.ndarray:
        n = len(metadata_list)
        scores = np.zeros(n, dtype=np.float32)

        for i, meta in enumerate(metadata_list):
            content = meta.get("content", "")
            score = 0.0

            chinese_chars = len([c for c in content if '\u4e00' <= c <= '\u9fff'])
            total_chars = max(len(content), 1)
            chinese_ratio = chinese_chars / total_chars
            score += 0.3 * min(chinese_ratio * 2, 1.0)

            sentences = len([c for c in content if c in '。！？；'])
            score += 0.3 * min(sentences / 5.0, 1.0)

            quality_weight = meta.get("quality_weight", 1.0)
            score += 0.2 * quality_weight

            length_score = min(len(content) / 300.0, 1.0)
            score += 0.2 * length_score

            scores[i] = score

        return scores

    def clean_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict],
                         dedup_threshold: float = VECTOR_DEDUP_THRESHOLD,
                         quality_filter_ratio: float = VECTOR_QUALITY_FILTER_RATIO) -> Tuple[np.ndarray, List[Dict]]:
        print(f"[VECTOR-CLEAN] 开始向量清洗: {len(embeddings)} 个原始向量")

        embeddings, metadata_list = self.dedup_embeddings(embeddings, metadata_list, dedup_threshold)
        print(f"[VECTOR-CLEAN] 冗余去重后: {len(embeddings)} 个向量")

        embeddings, metadata_list = self.quality_filter_embeddings(embeddings, metadata_list, quality_filter_ratio)
        print(f"[VECTOR-CLEAN] 质量筛选后: {len(embeddings)} 个向量")

        return embeddings, metadata_list

    def save(self, name="default"):
        import tempfile
        import shutil
        save_dir = os.path.join(VECTOR_STORE_DIR, name)
        os.makedirs(save_dir, exist_ok=True)
        index_path = os.path.join(save_dir, "faiss.index")
        meta_path = os.path.join(save_dir, "metadata.json")

        # FAISS C++底层不支持中文路径，先写到随机临时英文路径再复制
        tmp_dir = tempfile.mkdtemp(prefix="rag_faiss_tmp_")
        try:
            tmp_index = os.path.join(tmp_dir, "faiss.index")
            faiss.write_index(self.index, tmp_index)
            shutil.copy2(tmp_index, index_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.id_to_metadata, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 向量存储已保存至: {save_dir}")

    def save_embeddings(self, embeddings: np.ndarray, name="default"):
        save_dir = self._get_save_dir(name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "embeddings_raw.npy"), embeddings)

    def load_embeddings(self, name="default") -> np.ndarray:
        save_dir = self._get_save_dir(name)
        emb_path = os.path.join(save_dir, "embeddings_raw.npy")
        if os.path.exists(emb_path):
            return np.load(emb_path)
        raise FileNotFoundError(f"向量缓存不存在: {emb_path}")

    def _get_save_dir(self, name="default") -> str:
        return os.path.join(VECTOR_STORE_DIR, name)

    def load(self, name="default"):
        import tempfile
        import shutil
        save_dir = os.path.join(VECTOR_STORE_DIR, name)
        index_path = os.path.join(save_dir, "faiss.index")
        meta_path = os.path.join(save_dir, "metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"向量存储缓存不存在: {save_dir}")

        # FAISS C++底层不支持中文路径，先复制到随机临时英文路径再读取
        tmp_dir = tempfile.mkdtemp(prefix="rag_faiss_tmp_")
        try:
            tmp_index = os.path.join(tmp_dir, "faiss.index")
            shutil.copy2(index_path, tmp_index)
            self.index = faiss.read_index(tmp_index)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        with open(meta_path, "r", encoding="utf-8") as f:
            self.id_to_metadata = json.load(f)
        self.id_to_metadata = {int(k): v for k, v in self.id_to_metadata.items()}
        self.current_id = max(self.id_to_metadata.keys()) + 1 if self.id_to_metadata else 0
        print(f"[INFO] 从项目目录加载向量存储: {save_dir}")
