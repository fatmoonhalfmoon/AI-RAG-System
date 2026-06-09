import numpy as np
from typing import List, Dict

from src.utils.text_processing import tokenize, normalize_text, token_set


def _normalize_doc_name(name: str) -> str:
    import re
    name = name.replace("法則", "法则")
    name = re.sub(r'[-_：:，,·]', '', name)
    return name


def _loose_match(retrieved_docs: List[str], relevant_docs: List[str]) -> int:
    hits = 0
    for rel in relevant_docs:
        rel_norm = _normalize_doc_name(rel)
        for ret in retrieved_docs:
            ret_norm = _normalize_doc_name(ret)
            if ret_norm.startswith(rel_norm) or rel_norm.startswith(ret_norm):
                hits += 1
                break
            if rel_norm in ret_norm or ret_norm in rel_norm:
                hits += 1
                break
    return hits


class Layer1Metrics:
    @staticmethod
    def doc_recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = None) -> float:
        if not relevant_docs:
            return 1.0
        unique_retrieved = list(dict.fromkeys(retrieved_docs))
        if k:
            unique_retrieved = unique_retrieved[:k]
        hits = _loose_match(unique_retrieved, relevant_docs)
        return hits / len(relevant_docs)

    @staticmethod
    def doc_precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = None) -> float:
        if not retrieved_docs:
            return 0.0
        unique_retrieved = list(dict.fromkeys(retrieved_docs))
        if k:
            unique_retrieved = unique_retrieved[:k]
        hits = _loose_match(unique_retrieved, relevant_docs)
        return hits / len(unique_retrieved)

    @staticmethod
    def doc_hit_rate_at_k(results: List[Dict], k: int = 10) -> float:
        hits = 0
        for r in results:
            recall = Layer1Metrics.doc_recall_at_k(
                r["retrieved_docs"], r["relevant_docs"], k
            )
            if recall > 0:
                hits += 1
        return hits / len(results) if results else 0.0


def _extract_core_terms(text: str) -> List[str]:
    return tokenize(text)


class Layer2Metrics:
    @staticmethod
    def snippet_coverage(retrieved_text: str, answer_snippets: List[str]) -> float:
        if not answer_snippets:
            return 1.0
        hits = 0
        for snippet in answer_snippets:
            core_terms = _extract_core_terms(snippet)
            if not core_terms:
                continue
            matched = sum(1 for t in core_terms if t in retrieved_text)
            n = len(core_terms)
            if matched == 0:
                continue
            if n == 1:
                threshold = 1.0
            elif n == 2:
                threshold = 1.0
            elif n == 3:
                threshold = 2 / 3
            else:
                threshold = 0.6
            if matched >= 2 and matched / n >= threshold:
                hits += 1
        return hits / len(answer_snippets)

    @staticmethod
    def context_utilization(retrieved_chunks: List[Dict], relevant_chunk_ids: List[str]) -> float:
        total_chars = sum(len(c.get("content", "")) for c in retrieved_chunks)
        if total_chars == 0:
            return 0.0
        relevant_chars = sum(
            len(c.get("content", ""))
            for c in retrieved_chunks
            if c.get("chunk_id") in relevant_chunk_ids
        )
        return relevant_chars / total_chars

    @staticmethod
    def chunk_recall_at_k(retrieved_chunks: List[Dict], relevant_chunk_ids: List[str], k: int = None) -> float:
        if not relevant_chunk_ids:
            return 1.0
        chunk_ids = [c.get("chunk_id", "") for c in retrieved_chunks]
        if k:
            chunk_ids = chunk_ids[:k]
        hits = sum(1 for cid in relevant_chunk_ids if cid in chunk_ids)
        return hits / len(relevant_chunk_ids)

    @staticmethod
    def chunk_precision_at_k(retrieved_chunks: List[Dict], relevant_chunk_ids: List[str], k: int = None) -> float:
        if not retrieved_chunks:
            return 0.0
        chunk_ids = [c.get("chunk_id", "") for c in retrieved_chunks]
        if k:
            chunk_ids = chunk_ids[:k]
        hits = sum(1 for cid in chunk_ids if cid in relevant_chunk_ids)
        return hits / len(chunk_ids)


class Layer3Metrics:
    @staticmethod
    def mrr_at_k(results: List[Dict], k: int = 10) -> float:
        rr_sum = 0.0
        for r in results:
            retrieved_docs = r["retrieved_docs"][:k]
            relevant_docs = r["relevant_docs"]
            rank = 0
            for i, doc in enumerate(retrieved_docs, 1):
                if _loose_match([doc], relevant_docs) > 0:
                    rank = i
                    break
            if rank > 0:
                rr_sum += 1.0 / rank
        return rr_sum / len(results) if results else 0.0

    @staticmethod
    def ndcg_at_k(results: List[Dict], k: int = 10) -> float:
        ndcg_sum = 0.0
        for r in results:
            retrieved_docs = r["retrieved_docs"][:k]
            relevant_docs = r["relevant_docs"]
            dcg = 0.0
            for i, doc in enumerate(retrieved_docs):
                rel = 1 if _loose_match([doc], relevant_docs) > 0 else 0
                dcg += rel / np.log2(i + 2)
            ideal_rels = sorted([1] * min(len(relevant_docs), k) + [0] * max(0, k - len(relevant_docs)), reverse=True)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
            if idcg > 0:
                ndcg_sum += dcg / idcg
        return ndcg_sum / len(results) if results else 0.0


class Layer4Metrics:
    @staticmethod
    def keyword_completeness(merged_context: str, reference_answer: str) -> float:
        ref_keywords = set(tokenize(reference_answer))
        if not ref_keywords:
            return 0.0
        ctx_tokens = token_set(merged_context)
        hits = sum(1 for kw in ref_keywords if kw in ctx_tokens)
        return hits / len(ref_keywords)

    @staticmethod
    def format_check(result: Dict) -> Dict[str, bool]:
        checks = {
            "has_merged_context": bool(result.get("merged_context")),
            "has_source_docs": isinstance(result.get("source_docs"), list),
            "has_retrieved_chunks": isinstance(result.get("retrieved_chunks"), list),
            "has_qa_hits": isinstance(result.get("qa_hits"), list),
            "merged_context_length_ok": len(result.get("merged_context", "")) <= 4000,
            "chunks_non_empty": all(c.get("content") for c in result.get("retrieved_chunks", [])),
        }
        return checks
