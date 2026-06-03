import re
import json
import os
import numpy as np
from typing import List, Dict, Optional
from src.config import QA_STORE_DIR, QA_SEMANTIC_WEIGHT, QA_BM25_WEIGHT


class QAExtractor:
    DEFINITION_PATTERNS = [
        (re.compile(r'([^，。；！？\n]{2,20})是指([^。；！？\n]{10,200})[。；]'), "definition"),
        (re.compile(r'([^，。；！？\n]{2,20})定义为([^。；！？\n]{10,200})[。；]'), "definition"),
        (re.compile(r'所谓([^，。；！？\n]{2,20})[，,]就是([^。；！？\n]{10,200})[。；]'), "definition"),
        (re.compile(r'所谓([^，。；！？\n]{2,20})[，,]是指([^。；！？\n]{10,200})[。；]'), "definition"),
        (re.compile(r'([^，。；！？\n]{2,20})的核心是([^。；！？\n]{10,200})[。；]'), "definition"),
        (re.compile(r'([^，。；！？\n]{2,20})的本质是([^。；！？\n]{10,200})[。；]'), "definition"),
        (re.compile(r'([^，。；！？\n]{2,20})指的是([^。；！？\n]{10,200})[。；]'), "definition"),
    ]

    FRAMEWORK_PATTERNS = [
        (re.compile(r'([^，。；！？\n]{2,20})包括以下(?:几个)?(?:方面|部分|要素|维度|类型|阶段)[：:]([^。；！？\n]{15,300})[。；]'), "framework"),
        (re.compile(r'([^，。；！？\n]{2,20})主要(?:有|包括|分为|包含)[：:]?([^。；！？\n]{15,300})[。；]'), "framework"),
        (re.compile(r'([^，。；！？\n]{2,20})分为([^。；！？\n]{15,300})[。；]'), "framework"),
        (re.compile(r'([^，。；！？\n]{2,20})由([^。；！？\n]{5,30})组成'), "framework"),
        (re.compile(r'([^，。；！？\n]{2,20})由([^。；！？\n]{5,30})构成'), "framework"),
    ]

    PRINCIPLE_PATTERNS = [
        (re.compile(r'([^，。；！？\n]{2,20})的原则是([^。；！？\n]{10,200})[。；]'), "principle"),
        (re.compile(r'([^，。；！？\n]{2,20})应遵循([^。；！？\n]{10,200})[。；]'), "principle"),
        (re.compile(r'([^，。；！？\n]{2,20})的基本原则(?:包括|有)[：:]?([^。；！？\n]{10,200})[。；]'), "principle"),
    ]

    COMPARISON_PATTERNS = [
        (re.compile(r'([^，。；！？\n]{2,20})与([^，。；！？\n]{2,20})的区别(?:在于|是)[：:]?([^。；！？\n]{10,200})[。；]'), "comparison"),
        (re.compile(r'([^，。；！？\n]{2,20})不同于([^，。；！？\n]{2,20})[，,]([^。；！？\n]{10,200})[。；]'), "comparison"),
    ]

    QUESTION_TEMPLATES = {
        "definition": "什么是{subject}？",
        "framework": "{subject}包含哪些内容？",
        "principle": "{subject}的原则是什么？",
        "comparison": "{subject_a}和{subject_b}有什么区别？",
    }

    def extract_from_chunks(self, chunks: List[Dict]) -> List[Dict]:
        qa_pairs = []
        seen_questions = set()

        for chunk in chunks:
            content = chunk.get("content", "")
            source = chunk.get("source_doc", "")
            chapter = chunk.get("chapter", "")
            chunk_id = chunk.get("chunk_id", "")

            chunk_qa = []
            chunk_qa.extend(self._extract_by_patterns(content, self.DEFINITION_PATTERNS, "definition"))
            chunk_qa.extend(self._extract_by_patterns(content, self.FRAMEWORK_PATTERNS, "framework"))
            chunk_qa.extend(self._extract_by_patterns(content, self.PRINCIPLE_PATTERNS, "principle"))
            chunk_qa.extend(self._extract_comparison(content))

            for qa in chunk_qa:
                q_normalized = re.sub(r'\s+', '', qa["question"])
                if q_normalized in seen_questions:
                    continue
                seen_questions.add(q_normalized)
                qa_pairs.append({
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "qa_type": qa["qa_type"],
                    "source_doc": source,
                    "chapter": chapter,
                    "chunk_id": chunk_id,
                })

        return qa_pairs

    def _extract_by_patterns(self, text: str, patterns: List[tuple], qa_type: str) -> List[Dict]:
        results = []
        for pattern, _ in patterns:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) >= 2:
                    subject = groups[0].strip()
                    answer_text = groups[-1].strip()
                    if len(subject) < 2 or len(answer_text) < 10:
                        continue
                    question = self.QUESTION_TEMPLATES[qa_type].format(subject=subject)
                    results.append({
                        "question": question,
                        "answer": answer_text,
                        "qa_type": qa_type,
                    })
        return results

    def _extract_comparison(self, text: str) -> List[Dict]:
        results = []
        for pattern, _ in self.COMPARISON_PATTERNS:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) >= 3:
                    subject_a = groups[0].strip()
                    subject_b = groups[1].strip()
                    answer_text = groups[2].strip()
                    if len(subject_a) < 2 or len(subject_b) < 2 or len(answer_text) < 10:
                        continue
                    question = self.QUESTION_TEMPLATES["comparison"].format(
                        subject_a=subject_a, subject_b=subject_b
                    )
                    results.append({
                        "question": question,
                        "answer": answer_text,
                        "qa_type": "comparison",
                    })
        return results

    def save_qa_pairs(self, qa_pairs: List[Dict], filename: str = "qa_pairs.json") -> str:
        os.makedirs(QA_STORE_DIR, exist_ok=True)
        filepath = os.path.join(QA_STORE_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        return filepath

    def load_qa_pairs(self, filename: str = "qa_pairs.json") -> List[Dict]:
        filepath = os.path.join(QA_STORE_DIR, filename)
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
        from src.retriever import STOP_WORDS
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

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.qa_pairs:
            return []

        bm25_scores = self._bm25_search(query)
        semantic_scores = self._semantic_search(query)

        if semantic_scores is not None:
            combined = QA_SEMANTIC_WEIGHT * semantic_scores + QA_BM25_WEIGHT * bm25_scores
        else:
            combined = bm25_scores

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
        from src.retriever import STOP_WORDS
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
