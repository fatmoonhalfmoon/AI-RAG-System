"""
Pooling 候选池生成脚本
====================
对每道评估题，运行多套独立检索方法，合并候选池，导出 CSV 供人工标注。

核心思路：
  - 不依赖任何单一检索系统的输出作为金标
  - 用 BM25、Dense-Small、Dense-Large、QA 四路检索各自独立检索
  - 加上原始 relevant_chunk_ids 作为参考（标记为 Original-Gold）
  - 合并去重后导出 CSV，由人工判断每个 chunk 是否与 query 相关

用法：
  python scripts/generate_pool_for_annotation.py
  python scripts/generate_pool_for_annotation.py --top-k 30
  python scripts/generate_pool_for_annotation.py --output my_pool.csv
"""

import sys
import os
import json
import csv
import time
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from src.main import RAGPipeline
from src.retriever import BM25Retriever
from src.config import DENSE_TOP_K, BM25_TOP_K, QA_TOP_K


def build_chunk_lookup(pipeline: RAGPipeline) -> dict:
    """构建 chunk_id -> chunk 的查找表"""
    lookup = {}
    for chunk in pipeline.all_chunks:
        cid = chunk.get("chunk_id", "")
        if cid:
            lookup[cid] = chunk
    # 也从 small_dense_chunks 和 large_dense_chunks 补充
    for chunk in pipeline.retriever.small_dense_chunks:
        cid = chunk.get("chunk_id", "")
        if cid and cid not in lookup:
            lookup[cid] = chunk
    if pipeline.retriever.large_dense_chunks:
        for chunk in pipeline.retriever.large_dense_chunks:
            cid = chunk.get("chunk_id", "")
            if cid and cid not in lookup:
                lookup[cid] = chunk
    return lookup


def run_multi_retrieval(pipeline: RAGPipeline, query: str, top_k: int = 20) -> dict:
    """
    对单个 query 运行多套独立检索，返回每路检索结果。
    每路检索独立运行，不经过 RRF 融合/精排，保留原始检索视角。
    """
    retriever = pipeline.retriever
    results_by_source = {}

    # === 检索方法 1: BM25 纯词频检索 ===
    bm25_results = retriever._bm25_search(query, top_k=top_k)
    results_by_source["BM25"] = [
        {
            "chunk_id": cid,
            "score": float(score),
        }
        for cid, score, _ in bm25_results
    ]

    # === 检索方法 2: Dense-Small 向量检索 ===
    dense_small_results = retriever._dense_small_search(query, top_k=top_k)
    results_by_source["Dense-Small"] = [
        {
            "chunk_id": cid,
            "score": float(score),
        }
        for cid, score, _ in dense_small_results
    ]

    # === 检索方法 3: Dense-Large 向量检索 ===
    dense_large_results = retriever._dense_large_search(query, top_k=top_k)
    results_by_source["Dense-Large"] = [
        {
            "chunk_id": cid,
            "score": float(score),
        }
        for cid, score, _ in dense_large_results
    ]

    # === 检索方法 4: QA 知识点检索 ===
    qa_results = retriever._qa_search(query, top_k=min(top_k, QA_TOP_K))
    results_by_source["QA"] = [
        {
            "chunk_id": qa.get("chunk_id", ""),
            "score": float(qa.get("score", 0)),
        }
        for qa in qa_results
        if qa.get("chunk_id")
    ]

    return results_by_source


def generate_pool(dataset_path: str, output_csv: str, top_k: int = 20):
    """主流程：加载评估题 -> 多路检索 -> 合并候选池 -> 导出 CSV"""

    print("=" * 70)
    print("  Pooling 候选池生成 — 多系统检索 + 导出人工标注")
    print("=" * 70)

    # 1. 加载评估数据集
    print(f"\n[1/4] 加载评估数据集: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    print(f"  共 {len(questions)} 道评估题")

    # 2. 初始化 RAG 系统
    print(f"\n[2/4] 初始化 RAG 系统...")
    pipeline = RAGPipeline()
    pipeline.build_knowledge_base()

    # 构建 chunk 查找表
    chunk_lookup = build_chunk_lookup(pipeline)
    print(f"  chunk 查找表: {len(chunk_lookup)} 条")

    # 3. 对每道题运行多路检索，合并候选池
    print(f"\n[3/4] 对 {len(questions)} 道题运行多路检索 (每路 top-{top_k})...")
    pool_records = []
    stats = {
        "total_queries": len(questions),
        "total_candidates": 0,
        "unique_candidates": set(),
        "source_distribution": defaultdict(int),
        "missing_chunks": 0,
        "per_query_stats": [],
    }

    for i, q in enumerate(questions):
        query_id = q["query_id"]
        query = q["query"]
        original_gold_ids = set(q.get("relevant_chunk_ids", []))

        print(f"  [{i+1}/{len(questions)}] {query_id}: {query}")

        # 运行多路检索
        multi_results = run_multi_retrieval(pipeline, query, top_k=top_k)

        # 收集所有候选 chunk_id 及其来源
        candidate_sources = defaultdict(list)  # chunk_id -> [source_name, ...]
        candidate_scores = defaultdict(list)   # chunk_id -> [score, ...]

        for source_name, results in multi_results.items():
            for r in results:
                cid = r["chunk_id"]
                if cid:
                    candidate_sources[cid].append(source_name)
                    candidate_scores[cid].append(r["score"])
                    stats["source_distribution"][source_name] += 1

        # 加入原始金标中的 chunk（标记为 Original-Gold）
        for cid in original_gold_ids:
            if cid not in candidate_sources:
                candidate_sources[cid].append("Original-Gold")
                candidate_scores[cid].append(0)
                stats["source_distribution"]["Original-Gold"] += 1

        # 生成候选池记录
        query_candidates = 0
        for cid, sources in candidate_sources.items():
            chunk = chunk_lookup.get(cid, {})
            content = chunk.get("content", "")
            source_doc = chunk.get("source_doc", "")
            chapter = chunk.get("chapter", "")

            if not content:
                content = f"[CHUNK NOT FOUND IN KNOWLEDGE BASE]"
                stats["missing_chunks"] += 1

            # 截断内容预览，避免 CSV 过大
            content_preview = content[:300] if content else ""

            pool_records.append({
                "query_id": query_id,
                "query": query,
                "chunk_id": cid,
                "source_doc": source_doc,
                "chapter": chapter,
                "content_preview": content_preview,
                "retrieval_sources": "|".join(sorted(set(sources))),
                "num_sources": len(set(sources)),
                "max_score": round(max(candidate_scores[cid]), 4),
                "in_original_gold": 1 if cid in original_gold_ids else 0,
                # === 以下列由人工填写 ===
                "is_relevant": "",
                "relevance_level": "",  # 高/中/低
                "annotator": "",
                "notes": "",
            })

            stats["unique_candidates"].add(cid)
            query_candidates += 1

        stats["per_query_stats"].append({
            "query_id": query_id,
            "num_candidates": query_candidates,
            "num_sources": {s: len([r for r in multi_results.get(s, [])]) for s in multi_results},
        })

        stats["total_candidates"] += query_candidates

    # 4. 导出 CSV
    print(f"\n[4/4] 导出候选池到: {output_csv}")
    fieldnames = [
        "query_id", "query", "chunk_id", "source_doc", "chapter",
        "content_preview", "retrieval_sources", "num_sources", "max_score",
        "in_original_gold",
        "is_relevant", "relevance_level", "annotator", "notes",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pool_records)

    # 5. 保存统计信息
    stats_path = output_csv.replace(".csv", "_stats.json")
    stats["unique_candidates"] = len(stats["unique_candidates"])
    stats["source_distribution"] = dict(stats["source_distribution"])

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 6. 打印汇总
    print(f"\n{'='*70}")
    print(f"  Pooling 候选池生成完成")
    print(f"{'='*70}")
    print(f"  评估题数:        {stats['total_queries']}")
    print(f"  候选记录总数:    {stats['total_candidates']}")
    print(f"  去重 chunk 数:   {stats['unique_candidates']}")
    print(f"  平均每题候选数:  {stats['total_candidates'] / max(stats['total_queries'], 1):.1f}")
    print(f"  缺失 chunk 数:   {stats['missing_chunks']}")
    print(f"")
    print(f"  各检索来源分布:")
    for source, count in sorted(stats["source_distribution"].items()):
        print(f"    {source}: {count}")
    print(f"")
    print(f"  CSV 文件: {output_csv}")
    print(f"  统计文件: {stats_path}")
    print(f"")
    print(f"  下一步: 请在 CSV 中填写 is_relevant 列")
    print(f"    is_relevant = 1 表示该 chunk 与 query 相关")
    print(f"    is_relevant = 0 表示该 chunk 与 query 不相关")
    print(f"  填写完成后运行:")
    print(f"    python scripts/import_annotations.py --csv {output_csv}")
    print(f"{'='*70}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pooling 候选池生成 — 多系统检索 + 导出人工标注")
    parser.add_argument("--dataset", type=str, default="eval_data/eval_dataset.json",
                        help="评估数据集路径")
    parser.add_argument("--output", type=str, default="eval_data/annotation_pool.csv",
                        help="输出 CSV 路径")
    parser.add_argument("--top-k", type=int, default=20,
                        help="每路检索返回的 top-K 数量（默认20）")
    args = parser.parse_args()

    generate_pool(args.dataset, args.output, args.top_k)
