import os
import json
import numpy as np
from typing import List, Dict, Optional
from src.core.config import QA_STORE_DIR, QA_SEMANTIC_WEIGHT, QA_BM25_WEIGHT


class QAExtractor:
    """QA对管理器：负责QA对的加载与保存。

    注：本系统已弃用基于正则的自动提取逻辑。所有QA对由专用子代理
    通读全文后利用大模型能力生成，确保问题与答案高质量且严格对应。
    """

    def __init__(self):
        pass

    def extract_from_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """不再从chunk中自动提取，若已有手工/AI生成的QA文件则加载之。"""
        existing = self.load_qa_pairs()
        if existing:
            print(f"[QAExtractor] 发现已存在的QA对文件，共 {len(existing)} 对，跳过自动生成。")
            return existing
        print("[QAExtractor] 暂无预生成QA对，返回空列表。请运行QA生成流程补充。")
        return []

    def save_qa_pairs(self, qa_pairs: List[Dict], filename: str = "qa_pairs_final.json") -> str:
        os.makedirs(QA_STORE_DIR, exist_ok=True)
        safe_name = os.path.basename(filename)
        filepath = os.path.join(QA_STORE_DIR, safe_name)

        import datetime
        enriched = []
        for qa in qa_pairs:
            item = qa.copy()
            if "gold_source" not in item:
                item["gold_source"] = "ai_generated"
            item["generated_by"] = "ai_subagent_v1"
            item["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
            enriched.append(item)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)
        return filepath

    def load_qa_pairs(self, filename: str = "qa_pairs_final.json") -> List[Dict]:
        safe_name = os.path.basename(filename)
        filepath = os.path.join(QA_STORE_DIR, safe_name)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return []


class QARetriever:
    def __init__(self, qa_pairs: List[Dict], embedding_model=None):
        self.qa_pairs = qa_pairs
        self.embedding_model = embedding_model
        self._question_embeddings = None
        self._build_index()

    def _build_index(self):
        import jieba
        from src.utils.constants import STOP_WORDS
        self.question_tokens = []
        for qa in self.qa_pairs:
            tokens = [w for w in jieba.cut(qa["question"]) if len(w.strip()) >= 2 and w.strip() not in STOP_WORDS]
            self.question_tokens.append(tokens)

        if self.embedding_model is not None and len(self.qa_pairs) > 0:
            questions = [qa["question"] for qa in self.qa_pairs]
            print(f"[QA-RETRIEVER] 使用BGE语义向量化 {len(questions)} 个问题")
            self._question_embeddings = self.embedding_model.encode_documents(
                questions, batch_size=64, show_progress=False
            )
            norms = np.linalg.norm(self._question_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._question_embeddings = (self._question_embeddings / norms).astype(np.float32)
            print(f"[QA-RETRIEVER] 问题向量维度: {self._question_embeddings.shape}")
        else:
            print(f"[QA-RETRIEVER] 无Embedding模型，使用纯BM25检索")

    def search(self, query: str, top_k: int = 10,
               query_type: str = None, comparison_entities: List[str] = None) -> List[Dict]:
        if not self.qa_pairs:
            return []

        bm25_scores = self._bm25_search(query)
        semantic_scores = self._semantic_search(query)

        if semantic_scores is not None:
            combined = QA_SEMANTIC_WEIGHT * semantic_scores + QA_BM25_WEIGHT * bm25_scores
        else:
            combined = bm25_scores

        if query_type == "comparison" and comparison_entities and len(comparison_entities) >= 2:
            for idx, qa in enumerate(self.qa_pairs):
                qa_text = qa.get("question", "") + qa.get("answer", "")
                hits = sum(1 for e in comparison_entities if len(e) >= 2 and e in qa_text)
                if hits >= 2:
                    combined[idx] *= 1.6
                elif hits == 1:
                    combined[idx] *= 1.15
                if qa.get("qa_type") == "comparison":
                    combined[idx] *= 1.25
        elif query_type:
            type_map = {"definition": "definition", "framework": "framework",
                        "application": "application", "comparison": "comparison",
                        "method": "framework", "cause": "principle"}
            target = type_map.get(query_type, query_type)
            for idx, qa in enumerate(self.qa_pairs):
                if qa.get("qa_type") == target:
                    combined[idx] *= 1.2

        top_indices = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if combined[idx] > 0:
                qa = self.qa_pairs[idx].copy()
                qa["score"] = float(combined[idx])
                qa["bm25_score"] = float(bm25_scores[idx])
                if semantic_scores is not None:
                    qa["semantic_score"] = float(semantic_scores[idx])
                results.append(qa)
        return results

    def _bm25_search(self, query: str) -> np.ndarray:
        import jieba
        from src.utils.constants import STOP_WORDS
        from rank_bm25 import BM25Okapi

        query_tokens = [w for w in jieba.cut(query) if len(w.strip()) >= 2 and w.strip() not in STOP_WORDS]
        if not query_tokens:
            return np.zeros(len(self.qa_pairs), dtype=np.float32)

        bm25 = BM25Okapi(self.question_tokens)
        scores = bm25.get_scores(query_tokens)
        scores = np.array(scores, dtype=np.float32)
        max_s = scores.max()
        if max_s > 0:
            scores = scores / max_s
        return scores

    def _semantic_search(self, query: str) -> Optional[np.ndarray]:
        if self._question_embeddings is None or self.embedding_model is None:
            return None

        query_emb = self.embedding_model.encode_query(query)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        query_emb = query_emb.astype(np.float32)

        sims = self._question_embeddings @ query_emb
        max_s = sims.max()
        if max_s > 0:
            sims = sims / max_s
        return sims
