import jieba
import numpy as np
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from src.core.config import (
    DENSE_TOP_K, BM25_TOP_K, QA_TOP_K, RRF_K, MMR_LAMBDA,
    FINAL_TOP_K, MAX_CHUNKS_PER_DOC, QUERY_EXPAND_COUNT,
    DENSE_SMALL_WEIGHT, DENSE_LARGE_WEIGHT, BM25_WEIGHT, QA_WEIGHT,
    USE_CROSS_ENCODER, CROSS_ENCODER_MODEL, CROSS_ENCODER_TOP_K, CROSS_ENCODER_BATCH_SIZE,
)
from src.indexing.embedding import EmbeddingModel
from src.indexing.vector_store import FAISSVectorStore

from src.utils.constants import STOP_WORDS, QUESTION_TYPE_PATTERNS

DOMAIN_SYNONYMS = {
    "核心思想": ["管理理念", "管理原理", "管理基本原则", "管理学核心", "根本思想"],
    "核心": ["基本原理", "核心理念", "基础理论", "根本", "关键"],
    "思想": ["理论", "观点", "理念", "学说", "主张"],
    "战略": ["战略管理", "竞争策略", "战略规划", "战略思维"],
    "领导": ["领导力", "管理风格", "领导方式", "领导理论"],
    "领导力": ["领导", "管理风格", "领导方式", "领导理论", "领导行为", "影响力"],
    "组织": ["组织结构", "组织管理", "组织设计", "组织行为"],
    "人力": ["人力资源", "员工管理", "人事管理", "人才管理"],
    "营销": ["市场营销", "营销策略", "市场推广", "营销管理"],
    "财务": ["财务管理", "资金管理", "财务决策", "财务分析"],
    "运营": ["运营管理", "生产管理", "运作管理", "流程管理"],
    "沟通": ["管理沟通", "交流技巧", "信息传递", "协调"],
    "文化": ["组织文化", "企业文化", "价值观", "文化氛围"],
    "创新": ["创新管理", "变革管理", "组织变革", "持续创新"],
    "决策": ["管理决策", "决策理论", "决策过程", "决策方法"],
    "目标": ["组织目标", "目标管理", "战略目标", "绩效目标"],
    "管理": ["管理学", "管理理论", "管理方法", "管理实践", "经营管理"],
    "理论": ["学说", "原理", "方法论", "框架", "模型"],
    "分析": ["评估", "研究", "考察", "诊断"],
    "方法": ["方式", "手段", "途径", "工具", "技术"],
    "管理类": ["管理学", "管理学知识", "管理学科"],
    "霍桑": ["梅奥", "人际关系", "行为科学", "非正式组织"],
    "SWOT": ["优势劣势机会威胁", "战略分析", "态势分析"],
    "波特": ["五力模型", "竞争分析", "行业结构", "竞争战略"],
    "4P": ["营销组合", "产品价格渠道促销", "营销策略"],
    "精益": ["精益生产", "丰田", "JIT", "准时制", "消除浪费"],
    "法约尔": ["一般管理", "管理职能", "十四条", "管理原则"],
    "OKR": ["目标与关键结果", "关键结果", "目标管理", "OKR工作法"],
    "德鲁克": ["卓有成效", "知识工作者", "目标管理", "管理实践", "现代管理学"],
    "华为": ["任正非", "华为管理法", "狼性文化", "以奋斗者为本"],
    "小团队": ["小团队管理", "团队管理", "主管", "基层管理"],
    "基业长青": ["愿景", "核心价值观", "造钟", "高瞻远瞩", "长青企业", "持久成功", "卓越公司"],
    "8020": ["帕累托", "二八法则", "关键少数", "80/20", "高效管理", "科克", "效率法则"],
    "跨越": ["关键跨越", "新手主管", "管理者转型", "业务高手"],
    "效能": ["效率", "卓有成效", "时间管理", "要事优先"],
    "卓有成效": ["德鲁克", "管理者", "知识工作者", "时间管理", "决策", "有效性"],
    "执行力": ["管理思维", "团队执行", "落实", "行动力"],
    "泰勒": ["科学管理", "时间研究", "动作研究", "泰罗", "管理思想史", "雷恩"],
    "精益创业": ["精益", "创业管理", "MVP", "最小可行产品", "转型", "开发测量认知"],
    "创业": ["精益创业", "创业管理", "创新", "增长", "商业模式"],
    "管理思想史": ["管理学史", "管理理论演变", "泰罗", "法约尔", "人际关系学派", "雷恩", "贝德安"],
    "变革": ["组织变革", "变革管理", "创新管理", "持续改进"],
    "科学管理": ["泰勒制", "泰罗", "时间动作", "标准化", "工时定额", "管理思想史"],
    "公共管理": ["行政学", "公共行政", "官僚制", "韦伯", "新公共管理"],
    "竞争": ["竞争战略", "竞争优势", "竞争分析", "行业竞争"],
    "产品": ["产品思维", "产品管理", "产品设计", "用户体验"],
    "激励": ["动机", "需求", "奖励", "驱动力", "马斯洛"],
    "规划": ["战略规划", "计划", "目标设定", "长期规划"],
    "绩效": ["绩效考核", "绩效管理", "KPI", "评估"],
    "团队": ["团队管理", "团队建设", "协作", "团队合作"],
    "控制": ["管理控制", "监控", "反馈", "纠偏"],
    "计划": ["规划", "目标设定", "战略计划", "行动计划"],
    "组织行为": ["组织行为学", "个体行为", "群体行为", "组织文化"],
    "市场营销": ["营销管理", "市场分析", "消费者行为", "营销战略"],
    "人力资源": ["HRM", "员工管理", "招聘", "培训", "薪酬"],
    "知识管理": ["知识经济", "知识工作者", "学习型组织"],
    "质量管理": ["TQM", "六西格玛", "持续改进", "戴明"],
    "项目管理": ["项目规划", "项目执行", "项目控制", "PMBOK"],
    "管理者": ["卓有成效", "管理实践", "德鲁克", "主管", "领导"],
    "长青": ["基业长青", "持久", "可持续发展", "核心价值观"],
    "法則": ["法则", "8020", "帕累托", "二八"],
    "管理史": ["管理思想史", "管理学发展", "管理理论演变"],
}


class BM25Retriever:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        tokenized_corpus = [self._tokenize(c["content"]) for c in chunks]
        self.doc_name_tokens = [self._tokenize(c.get("source_doc", "")) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        return [w for w in jieba.cut(text) if w.strip() and w not in STOP_WORDS]

    def search(self, query: str, top_k=BM25_TOP_K) -> List[Tuple[int, float]]:
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        score_max = scores.max()
        if score_max > 0:
            scores = scores / score_max

        query_set = set(tokenized_query)
        for i, name_tokens in enumerate(self.doc_name_tokens):
            name_set = set(name_tokens)
            overlap = len(query_set & name_set)
            if overlap > 0:
                boost = 0.3 * overlap / max(len(name_set), 1)
                scores[i] = min(scores[i] + boost, 1.5)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


class CrossEncoderReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        import os
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ['HF_HUB_OFFLINE'] = '1'
        from sentence_transformers import CrossEncoder
        print(f"[CROSS-ENCODER] 加载精排模型: {self.model_name}")
        self._model = CrossEncoder(self.model_name, local_files_only=True)
        print(f"[CROSS-ENCODER] 模型加载完成")

    def rerank(self, query: str, chunks: List[Dict], top_k: int = FINAL_TOP_K) -> List[Dict]:
        if not chunks:
            return chunks

        self._load_model()

        pairs = [(query, c.get("content", "")) for c in chunks]

        # 批量处理避免内存溢出
        batch_size = CROSS_ENCODER_BATCH_SIZE
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            scores = self._model.predict(batch, show_progress_bar=False)
            all_scores.extend(scores)

        for i, c in enumerate(chunks):
            c["cross_encoder_score"] = float(all_scores[i])

        chunks.sort(key=lambda x: x.get("cross_encoder_score", 0), reverse=True)
        return chunks[:top_k]


class HybridRetriever:
    def __init__(self, embedding_model: EmbeddingModel,
                 small_vector_store: FAISSVectorStore,
                 large_vector_store: Optional[FAISSVectorStore],
                 all_chunks: List[Dict],
                 small_dense_chunks: List[Dict],
                 large_dense_chunks: Optional[List[Dict]],
                 small_dense_embeddings: np.ndarray = None,
                 large_dense_embeddings: np.ndarray = None,
                 qa_retriever=None):
        self.embedding_model = embedding_model
        self.small_vector_store = small_vector_store
        self.large_vector_store = large_vector_store
        self.all_chunks = all_chunks
        self.small_dense_chunks = small_dense_chunks
        self.large_dense_chunks = large_dense_chunks or []
        self.bm25 = BM25Retriever(all_chunks)
        self.qa_retriever = qa_retriever
        self._all_chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(all_chunks)}
        self._small_chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(small_dense_chunks)}
        self._large_chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(self.large_dense_chunks)} if large_dense_chunks else {}
        self._doc_names = list(set(c.get("source_doc", "") for c in all_chunks))
        self._small_embeddings_cache = small_dense_embeddings
        self._large_embeddings_cache = large_dense_embeddings
        self._cross_encoder = CrossEncoderReranker() if USE_CROSS_ENCODER else None

    def _dense_small_search(self, query: str, top_k=DENSE_TOP_K) -> List[Tuple[str, float, Dict]]:
        query_emb = self.embedding_model.encode_query(query)
        results = self.small_vector_store.search(query_emb, top_k)
        return [(meta.get("chunk_id", f"chunk_{idx}"), score, meta) for idx, score, meta in results]

    def _dense_large_search(self, query: str, top_k=DENSE_TOP_K) -> List[Tuple[str, float, Dict]]:
        if self.large_vector_store is None:
            return []
        query_emb = self.embedding_model.encode_query(query)
        results = self.large_vector_store.search(query_emb, top_k)
        return [(meta.get("chunk_id", f"chunk_{idx}"), score, meta) for idx, score, meta in results]

    def _bm25_search(self, query: str, top_k=BM25_TOP_K) -> List[Tuple[str, float, Dict]]:
        results = self.bm25.search(query, top_k)
        bm25_results = []
        query_terms = set(self.bm25._tokenize(query))
        for idx, score in results:
            if 0 <= idx < len(self.all_chunks):
                chunk = self.all_chunks[idx]
                # 章节标题匹配boost
                chapter = chunk.get("chapter", "")
                if chapter:
                    chapter_terms = set(self.bm25._tokenize(chapter))
                    overlap = len(query_terms & chapter_terms)
                    if overlap > 0:
                        boost = 0.2 * overlap / max(len(chapter_terms), 1)
                        score = min(score * (1 + boost), 1.5)
                bm25_results.append((chunk["chunk_id"], score, chunk))
        return bm25_results

    def _qa_search(self, query: str, top_k=QA_TOP_K) -> List[Dict]:
        if self.qa_retriever is None:
            return []
        return self.qa_retriever.search(query, top_k)

    def _rrf_fusion(self, dense_small_results: List[Tuple[str, float, Dict]],
                     dense_large_results: List[Tuple[str, float, Dict]],
                     bm25_results: List[Tuple[str, float, Dict]],
                     qa_results: List[Dict] = None,
                     k=RRF_K) -> List[Tuple[str, float, Dict]]:
        rrf_scores: Dict[str, float] = {}
        meta_map: Dict[str, Dict] = {}
        dense_small_map: Dict[str, float] = {}
        dense_large_map: Dict[str, float] = {}
        bm25_map: Dict[str, float] = {}
        match_types: Dict[str, List[str]] = {}

        for rank, (chunk_id, score, meta) in enumerate(dense_small_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + DENSE_SMALL_WEIGHT / (k + rank + 1)
            meta_map[chunk_id] = meta
            dense_small_map[chunk_id] = score
            match_types.setdefault(chunk_id, []).append("dense_small")

        for rank, (chunk_id, score, meta) in enumerate(dense_large_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + DENSE_LARGE_WEIGHT / (k + rank + 1)
            if chunk_id not in meta_map:
                meta_map[chunk_id] = meta
            dense_large_map[chunk_id] = score
            match_types.setdefault(chunk_id, []).append("dense_large")

        for rank, (chunk_id, score, meta) in enumerate(bm25_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + BM25_WEIGHT / (k + rank + 1)
            if chunk_id not in meta_map:
                meta_map[chunk_id] = meta
            bm25_map[chunk_id] = score
            match_types.setdefault(chunk_id, []).append("bm25")

        if qa_results:
            for rank, qa in enumerate(qa_results):
                chunk_id = qa.get("chunk_id", "")
                if chunk_id:
                    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + QA_WEIGHT / (k + rank + 1)
                    if chunk_id not in meta_map:
                        for c in self.all_chunks:
                            if c["chunk_id"] == chunk_id:
                                meta_map[chunk_id] = c
                                break
                    match_types.setdefault(chunk_id, []).append("qa")

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            (cid, rrf_scores[cid], meta_map[cid],
             dense_small_map.get(cid, 0), dense_large_map.get(cid, 0),
             bm25_map.get(cid, 0),
             "+".join(match_types.get(cid, [])))
            for cid, _ in sorted_ids if cid in meta_map
        ]

    def _dedup_by_source(self, candidates: List[Tuple], max_per_doc=MAX_CHUNKS_PER_DOC) -> List[Tuple]:
        doc_counts: Dict[str, int] = {}
        result = []
        for item in candidates:
            chunk_id, rrf_score, meta, ds_score, dl_score, b_score, match_type = item
            source = meta.get("source_doc", "")
            count = doc_counts.get(source, 0)
            if count < max_per_doc:
                result.append(item)
                doc_counts[source] = count + 1
        return result

    def _get_mmr_lambda(self, query: str) -> float:
        """根据问题类型动态调整MMR的lambda参数"""
        if any(kw in query for kw in QUESTION_TYPE_PATTERNS["comparison"]):
            return 0.30  # 对比类问题需要更高多样性
        elif any(kw in query for kw in QUESTION_TYPE_PATTERNS["definition"]):
            return 0.60  # 定义类问题需要更高相关性
        return MMR_LAMBDA  # 默认

    def _mmr_rerank(self, candidates: List[Tuple],
                    query_embedding: np.ndarray,
                    lambda_param=MMR_LAMBDA,
                    top_k=FINAL_TOP_K) -> List[Tuple]:
        if len(candidates) <= top_k:
            return candidates

        candidate_embeddings = []
        candidate_items = []
        for item in candidates:
            chunk_id, rrf_score, meta, ds_score, dl_score, b_score, match_type = item
            candidate_items.append(item)
            idx = self._small_chunk_id_to_idx.get(chunk_id)
            if idx is not None and self._small_embeddings_cache is not None and idx < len(self._small_embeddings_cache):
                candidate_embeddings.append(self._small_embeddings_cache[idx])
            else:
                idx_l = self._large_chunk_id_to_idx.get(chunk_id)
                if idx_l is not None and self._large_embeddings_cache is not None and idx_l < len(self._large_embeddings_cache):
                    candidate_embeddings.append(self._large_embeddings_cache[idx_l])
                else:
                    emb = self.embedding_model.encode_documents(
                        [meta.get("content", "")], show_progress=False
                    )
                    candidate_embeddings.append(emb[0])

        if not candidate_embeddings:
            return candidates[:top_k]

        emb_matrix = np.array(candidate_embeddings, dtype=np.float32)
        query_emb = query_embedding.reshape(1, -1).astype(np.float32)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        emb_matrix = emb_matrix / norms
        sim_to_query = np.dot(emb_matrix, query_emb.T).flatten()

        mmr_selected = []
        unselected = list(range(len(candidate_items)))

        for _ in range(min(top_k, len(candidate_items))):
            mmr_scores = []
            for i in unselected:
                relevance = lambda_param * sim_to_query[i]
                if mmr_selected:
                    sim_to_selected = np.dot(
                        emb_matrix[i:i + 1], emb_matrix[mmr_selected].T
                    ).flatten()
                    max_sim = np.max(sim_to_selected)
                    diversity = (1 - lambda_param) * max_sim
                else:
                    diversity = 0
                mmr_scores.append(relevance - diversity)
            best_local = int(np.argmax(mmr_scores))
            best_global = unselected[best_local]
            mmr_selected.append(best_global)
            unselected.remove(best_global)

        return [candidate_items[i] for i in mmr_selected]

    def _final_rerank(self, mmr_results: List[Tuple],
                      query_embedding: np.ndarray) -> List[Tuple]:
        if len(mmr_results) <= 1:
            return mmr_results

        rrf_scores = np.array([item[1] for item in mmr_results])
        rrf_norm = rrf_scores / rrf_scores.max() if rrf_scores.max() > 0 else rrf_scores

        sim_scores = []
        for item in mmr_results:
            chunk_id, _, meta, _, _, _, _ = item
            idx = self._small_chunk_id_to_idx.get(chunk_id)
            if idx is not None and self._small_embeddings_cache is not None and idx < len(self._small_embeddings_cache):
                emb = self._small_embeddings_cache[idx].reshape(1, -1).astype(np.float32)
            else:
                idx_l = self._large_chunk_id_to_idx.get(chunk_id)
                if idx_l is not None and self._large_embeddings_cache is not None and idx_l < len(self._large_embeddings_cache):
                    emb = self._large_embeddings_cache[idx_l].reshape(1, -1).astype(np.float32)
                else:
                    emb = self.embedding_model.encode_documents(
                        [meta.get("content", "")], show_progress=False
                    ).reshape(1, -1).astype(np.float32)
            n = np.linalg.norm(emb)
            if n > 0:
                emb = emb / n
            sim = float(np.dot(emb.flatten(), query_embedding.flatten().astype(np.float32)))
            sim_scores.append(sim)
        sim_scores = np.array(sim_scores)

        combined = 0.35 * rrf_norm + 0.65 * sim_scores
        sorted_idx = np.argsort(combined)[::-1]
        return [mmr_results[i] for i in sorted_idx]

    def _generate_query_variants(self, query: str) -> List[str]:
        variants = [query]

        keywords = [w for w in jieba.cut(query)
                    if len(w.strip()) >= 2 and w.strip() not in STOP_WORDS]
        if len(keywords) >= 2:
            variants.append(" ".join(keywords))

        expanded_keywords_map: Dict[str, str] = {}
        for domain_word, synonyms in DOMAIN_SYNONYMS.items():
            if domain_word in query:
                for syn in synonyms[:3]:
                    if syn not in query and syn not in expanded_keywords_map:
                        expanded_keywords_map[syn] = domain_word

        if expanded_keywords_map:
            syn_list = list(expanded_keywords_map.keys())
            variants.append(query + " " + " ".join(syn_list[:4]))
            for syn in syn_list[:3]:
                variant = query.replace(
                    next((k for k, v in expanded_keywords_map.items() if v == expanded_keywords_map.get(syn)), ""), syn
                ) if syn in expanded_keywords_map else query
                if variant != query and variant not in variants:
                    variants.append(variant)

        if len(keywords) >= 3:
            shortened = " ".join(keywords[:len(keywords) // 2 + 1])
            if shortened not in variants:
                variants.append(shortened)

        for kw in keywords[:3]:
            if kw in DOMAIN_SYNONYMS:
                for syn in DOMAIN_SYNONYMS[kw][:2]:
                    if syn not in query:
                        v = kw + " " + syn
                        if v not in variants:
                            variants.append(v)

        if not any(w in query for w in ["管理", "理论", "方法"]):
            domain_boost = query + " 管理"
            if domain_boost not in variants:
                variants.append(domain_boost)

        for doc_name in self._doc_names:
            doc_tokens = set(jieba.cut(doc_name))
            query_tokens = set(keywords)
            overlap = query_tokens & doc_tokens - STOP_WORDS
            if len(overlap) >= 2 or (len(overlap) >= 1 and len(keywords) <= 3):
                if doc_name not in variants:
                    variants.append(doc_name)

        # 根据问题类型增加特定扩展
        for qtype, keywords_list in QUESTION_TYPE_PATTERNS.items():
            if any(kw in query for kw in keywords_list):
                if qtype == "definition":
                    v = query + " 概念 定义 含义"
                    if v not in variants:
                        variants.append(v)
                elif qtype == "comparison":
                    v = query + " 异同 对比分析 区别"
                    if v not in variants:
                        variants.append(v)
                elif qtype == "method":
                    v = query + " 步骤 流程 方法"
                    if v not in variants:
                        variants.append(v)
                elif qtype == "cause":
                    v = query + " 原因 因素 影响"
                    if v not in variants:
                        variants.append(v)
                break

        return variants[:QUERY_EXPAND_COUNT + 2]

    def _merge_adjacent_chunks(self, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return chunks

        by_source: Dict[str, List[Dict]] = {}
        for c in chunks:
            src = c.get("source_doc", "")
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(c)

        merged = []
        for src, src_chunks in by_source.items():
            src_chunks.sort(key=lambda x: x.get("chunk_index", 0))

            groups = []
            current_group = [src_chunks[0]]

            for i in range(1, len(src_chunks)):
                prev_idx = src_chunks[i - 1].get("chunk_index", 0)
                curr_idx = src_chunks[i].get("chunk_index", 0)
                if curr_idx - prev_idx <= 2:
                    current_group.append(src_chunks[i])
                else:
                    groups.append(current_group)
                    current_group = [src_chunks[i]]
            groups.append(current_group)

            for group in groups:
                if len(group) == 1:
                    merged.append(group[0])
                else:
                    combined_content = "\n".join(c.get("content", "") for c in group)
                    best_parent = max(group, key=lambda c: len(c.get("parent_content", "")))
                    merged.append({
                        **group[0],
                        "content": combined_content,
                        "parent_content": best_parent.get("parent_content", combined_content),
                        "merged_count": len(group),
                        "chunk_indices": [c.get("chunk_index", 0) for c in group],
                    })

        return merged

    def search(self, query: str,
               dense_top_k=DENSE_TOP_K,
               bm25_top_k=BM25_TOP_K,
               qa_top_k=QA_TOP_K,
               final_top_k=FINAL_TOP_K,
               use_query_expansion=True) -> Dict:
        query_variants = self._generate_query_variants(query) if use_query_expansion else [query]

        dense_small_score_map: Dict[str, float] = {}
        dense_small_meta_map: Dict[str, Dict] = {}
        dense_large_score_map: Dict[str, float] = {}
        dense_large_meta_map: Dict[str, Dict] = {}
        bm25_score_map: Dict[str, float] = {}
        bm25_meta_map: Dict[str, Dict] = {}

        for q in query_variants:
            for chunk_id, score, meta in self._dense_small_search(q, dense_top_k):
                if chunk_id not in dense_small_score_map or score > dense_small_score_map[chunk_id]:
                    dense_small_score_map[chunk_id] = score
                    dense_small_meta_map[chunk_id] = meta
            for chunk_id, score, meta in self._dense_large_search(q, dense_top_k):
                if chunk_id not in dense_large_score_map or score > dense_large_score_map[chunk_id]:
                    dense_large_score_map[chunk_id] = score
                    dense_large_meta_map[chunk_id] = meta
            for chunk_id, score, meta in self._bm25_search(q, bm25_top_k):
                if chunk_id not in bm25_score_map or score > bm25_score_map[chunk_id]:
                    bm25_score_map[chunk_id] = score
                    bm25_meta_map[chunk_id] = meta

        all_dense_small_results = [(cid, s, dense_small_meta_map[cid]) for cid, s in dense_small_score_map.items()]
        all_dense_large_results = [(cid, s, dense_large_meta_map[cid]) for cid, s in dense_large_score_map.items()]
        all_bm25_results = [(cid, s, bm25_meta_map[cid]) for cid, s in bm25_score_map.items()]

        qa_results = self._qa_search(query, qa_top_k)

        fused = self._rrf_fusion(all_dense_small_results, all_dense_large_results, all_bm25_results, qa_results)
        deduped = self._dedup_by_source(fused, MAX_CHUNKS_PER_DOC)

        query_embedding = self.embedding_model.encode_query(query)
        mmr_lambda = self._get_mmr_lambda(query)
        mmr_candidates = self._mmr_rerank(
            deduped, query_embedding, mmr_lambda, final_top_k
        )
        reranked = self._final_rerank(mmr_candidates, query_embedding)

        final_chunks_raw = []
        for item in reranked:
            chunk_id, rrf_score, meta, ds_score, dl_score, b_score, match_type = item
            final_chunks_raw.append({
                "rank": 0,
                "rrf_score": round(rrf_score, 6),
                "dense_small_score": round(ds_score, 4),
                "dense_large_score": round(dl_score, 4),
                "bm25_score": round(b_score, 4),
                "match_type": match_type,
                "chunk_id": chunk_id,
                "source_doc": meta.get("source_doc", ""),
                "content": meta.get("content", ""),
                "parent_content": meta.get("parent_content", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "chapter": meta.get("chapter", ""),
            })

        if self._cross_encoder is not None and len(final_chunks_raw) > 0:
            ce_top_k = min(CROSS_ENCODER_TOP_K, len(final_chunks_raw))
            final_chunks_raw = self._cross_encoder.rerank(query, final_chunks_raw, ce_top_k)

        final_chunks = self._merge_adjacent_chunks(final_chunks_raw)
        for i, c in enumerate(final_chunks):
            c["rank"] = i + 1

        qa_hits = []
        for qa in qa_results[:5]:
            qa_hits.append({
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "source_doc": qa.get("source_doc", ""),
                "qa_type": qa.get("qa_type", ""),
                "score": round(qa.get("score", 0), 4),
            })

        return {
            "query": query,
            "expanded_queries": query_variants,
            "retrieved_chunks": final_chunks,
            "qa_hits": qa_hits,
            "dense_small_candidates": len(dense_small_score_map),
            "dense_large_candidates": len(dense_large_score_map),
            "bm25_candidates": len(bm25_score_map),
            "qa_candidates": len(qa_results),
            "fused_candidates": len(fused),
            "deduped_candidates": len(deduped),
            "final_count": len(final_chunks),
        }
