"""
基于人工金标的评估脚本
====================
使用人工标注后的 eval_dataset_human.json 运行评估。
与 eval_full.py 的区别：
  - 强制检查 gold_source == "human"
  - 报告中显示金标来源与可信度
  - 区分严格模式（仅人工金标）和宽松模式（全部）

用法：
  python scripts/run_eval_with_human_gold.py
  python scripts/run_eval_with_human_gold.py --dataset eval_data/eval_dataset_human.json
  python scripts/run_eval_with_human_gold.py --mode strict
"""

import sys
import os
import time
import json
import argparse
from typing import List, Dict
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from src.main import RAGPipeline
from src.eval.metrics import Layer1Metrics, Layer2Metrics, Layer3Metrics, Layer4Metrics


def load_dataset(dataset_path: str, mode: str = "strict") -> List[Dict]:
    """加载评估数据集，根据模式过滤"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])

    # 检查金标来源分布
    human_count = sum(1 for q in questions if q.get("gold_source") == "human")
    pipeline_count = sum(1 for q in questions if q.get("gold_source") == "pipeline_generated")
    other_count = len(questions) - human_count - pipeline_count

    print(f"[数据集] 金标来源分布:")
    print(f"  人工标注 (human):              {human_count}")
    print(f"  流水线生成 (pipeline_generated): {pipeline_count}")
    print(f"  其他:                           {other_count}")
    print(f"  总计:                           {len(questions)}")

    if mode == "strict":
        filtered = [q for q in questions if q.get("gold_source") == "human"]
        excluded = len(questions) - len(filtered)
        if excluded > 0:
            print(f"\n[严格模式] 排除 {excluded} 道非人工标注题")
        questions = filtered
    elif mode == "semi":
        # 半严格：人工金标优先，pipeline 题标记为低可信度
        for q in questions:
            if q.get("gold_source") != "human":
                q["_low_confidence"] = True
        print(f"\n[半严格模式] 保留全部 {len(questions)} 题，非人工金标记为低可信度")

    if not questions:
        print("[错误] 没有可评估的题目！请先完成人工标注。")
        sys.exit(1)

    return questions


def run_evaluation(dataset_path: str, mode: str = "strict",
                   top_k_values: List[int] = None, force_rebuild: bool = False):
    """主评估流程"""

    if top_k_values is None:
        top_k_values = [3, 5, 8, 10, 15]

    print("=" * 70)
    print(f"  RAG 检索质量评估 — 人工金标模式 ({mode})")
    print("=" * 70)

    # 1. 加载数据集
    print(f"\n[1/5] 加载评估数据集: {dataset_path}")
    questions = load_dataset(dataset_path, mode)
    print(f"  实际评估题数: {len(questions)}")

    # 2. 初始化 RAG 系统
    print(f"\n[2/5] 初始化 RAG 系统...")
    pipeline = RAGPipeline()
    pipeline.build_knowledge_base(force_rebuild=force_rebuild)

    # 3. 检索评估
    print(f"\n[3/5] 对 {len(questions)} 题进行检索评估...")
    all_results = []
    for i, q in enumerate(questions):
        query = q["query"]
        print(f"  [{i+1}/{len(questions)}] {q['query_id']}: {query}")

        result = pipeline.search(query, top_k=max(top_k_values))

        all_results.append({
            "query_id": q["query_id"],
            "query": query,
            "query_type": q["query_type"],
            "difficulty": q["difficulty"],
            "relevant_docs": q["relevant_docs"],
            "answer_snippets": q["answer_snippets"],
            "reference_answer": q["reference_answer"],
            "relevant_chunk_ids": q.get("relevant_chunk_ids", []),
            "gold_source": q.get("gold_source", "unknown"),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "retrieved_docs": list(dict.fromkeys(
                c["source_doc"] for c in result.get("retrieved_chunks", [])
            )),
            "merged_context": result.get("merged_context", ""),
            "source_docs": result.get("source_docs", []),
            "qa_hits": result.get("qa_hits", []),
            "retrieval_time_ms": result.get("retrieval_time_ms", 0),
            "_low_confidence": q.get("_low_confidence", False),
        })

    # 4. 计算指标
    print(f"\n[4/5] 计算评估指标...")
    report = compute_metrics(all_results, top_k_values, mode)

    # 5. 保存报告
    print(f"\n[5/5] 保存报告...")
    mode_suffix = f"_{mode}" if mode != "strict" else ""
    report_path = os.path.join("output", f"eval_human_gold{mode_suffix}_report.json")
    os.makedirs("output", exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印控制台报告
    print_console_report(report)

    print(f"\n报告已保存: {report_path}")
    return report


def compute_metrics(results: List[Dict], top_k_values: List[int], mode: str) -> Dict:
    """计算全部指标"""
    report = {
        "version": "3.0",
        "mode": f"human_gold_{mode}",
        "total_questions": len(results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gold_source_stats": {
            "human": sum(1 for r in results if r.get("gold_source") == "human"),
            "pipeline_generated": sum(1 for r in results if r.get("gold_source") == "pipeline_generated"),
            "trust_level": "high" if all(r.get("gold_source") == "human" for r in results) else "mixed",
        },
        "layer1": {},
        "layer2": {},
        "layer3": {},
        "layer4": {},
        "group_stats": {},
        "difficulty_stats": {},
        "per_query_details": [],
    }

    # Layer 1: 文档级召回/精度
    for k in top_k_values:
        recalls = [Layer1Metrics.doc_recall_at_k(r["retrieved_docs"], r["relevant_docs"], k) for r in results]
        precisions = [Layer1Metrics.doc_precision_at_k(r["retrieved_docs"], r["relevant_docs"], k) for r in results]
        report["layer1"][f"doc_recall@{k}"] = round(sum(recalls) / len(recalls), 4) if recalls else 0
        report["layer1"][f"doc_precision@{k}"] = round(sum(precisions) / len(precisions), 4) if precisions else 0
    report["layer1"]["doc_hit_rate@10"] = Layer1Metrics.doc_hit_rate_at_k(results, 10)

    # Layer 2: 片段覆盖 & chunk 指标
    coverages = []
    chunk_recalls = []
    chunk_precisions = []
    for r in results:
        text = " ".join(c.get("parent_content", c.get("content", "")) for c in r["retrieved_chunks"])
        coverages.append(Layer2Metrics.snippet_coverage(text, r["answer_snippets"]))

        if r.get("relevant_chunk_ids"):
            chunk_recalls.append(Layer2Metrics.chunk_recall_at_k(r["retrieved_chunks"], r["relevant_chunk_ids"], 10))
            chunk_precisions.append(Layer2Metrics.chunk_precision_at_k(r["retrieved_chunks"], r["relevant_chunk_ids"], 10))

    report["layer2"]["snippet_coverage@10"] = round(sum(coverages) / len(coverages), 4) if coverages else 0
    report["layer2"]["chunk_recall@10"] = round(sum(chunk_recalls) / len(chunk_recalls), 4) if chunk_recalls else 0
    report["layer2"]["chunk_precision@10"] = round(sum(chunk_precisions) / len(chunk_precisions), 4) if chunk_precisions else 0

    # Layer 3: 排序质量
    for k in top_k_values:
        report["layer3"][f"mrr@{k}"] = round(Layer3Metrics.mrr_at_k(results, k), 4)
        report["layer3"][f"ndcg@{k}"] = round(Layer3Metrics.ndcg_at_k(results, k), 4)

    # Layer 4: 下游兼容性
    kw_completeness = [Layer4Metrics.keyword_completeness(r["merged_context"], r["reference_answer"]) for r in results]
    report["layer4"]["keyword_completeness"] = round(sum(kw_completeness) / len(kw_completeness), 4) if kw_completeness else 0

    # 按类型分组
    groups = defaultdict(lambda: {"n": 0, "recalls": [], "coverages": [], "mrrs": []})
    for r in results:
        t = r["query_type"]
        groups[t]["n"] += 1
        groups[t]["recalls"].append(Layer1Metrics.doc_recall_at_k(r["retrieved_docs"], r["relevant_docs"], 10))
        text = " ".join(c.get("parent_content", c.get("content", "")) for c in r["retrieved_chunks"])
        groups[t]["coverages"].append(Layer2Metrics.snippet_coverage(text, r["answer_snippets"]))
        groups[t]["mrrs"].append(Layer3Metrics.mrr_at_k([r], 10))

    for t, stats in groups.items():
        report["group_stats"][t] = {
            "n": stats["n"],
            "recall@10": round(sum(stats["recalls"]) / len(stats["recalls"]), 4),
            "coverage@10": round(sum(stats["coverages"]) / len(stats["coverages"]), 4),
            "mrr@10": round(sum(stats["mrrs"]) / len(stats["mrrs"]), 4),
        }

    # 按难度分组
    diffs = defaultdict(lambda: {"n": 0, "recalls": [], "coverages": []})
    for r in results:
        d = r["difficulty"]
        diffs[d]["n"] += 1
        diffs[d]["recalls"].append(Layer1Metrics.doc_recall_at_k(r["retrieved_docs"], r["relevant_docs"], 10))
        text = " ".join(c.get("parent_content", c.get("content", "")) for c in r["retrieved_chunks"])
        diffs[d]["coverages"].append(Layer2Metrics.snippet_coverage(text, r["answer_snippets"]))

    for d, stats in diffs.items():
        report["difficulty_stats"][d] = {
            "n": stats["n"],
            "recall@10": round(sum(stats["recalls"]) / len(stats["recalls"]), 4),
            "coverage@10": round(sum(stats["coverages"]) / len(stats["coverages"]), 4),
        }

    # 每题详情
    for r in results:
        recall = Layer1Metrics.doc_recall_at_k(r["retrieved_docs"], r["relevant_docs"], 10)
        text = " ".join(c.get("parent_content", c.get("content", "")) for c in r["retrieved_chunks"])
        coverage = Layer2Metrics.snippet_coverage(text, r["answer_snippets"])
        mrr = Layer3Metrics.mrr_at_k([r], 10)

        report["per_query_details"].append({
            "query_id": r["query_id"],
            "query": r["query"],
            "gold_source": r.get("gold_source", "unknown"),
            "doc_recall@10": round(recall, 4),
            "snippet_coverage@10": round(coverage, 4),
            "mrr@10": round(mrr, 4),
            "retrieval_time_ms": r.get("retrieval_time_ms", 0),
        })

    return report


def print_console_report(report: Dict):
    """打印控制台报告"""
    print(f"\n{'='*70}")
    print(f"  评估结果 (金标来源: {report['gold_source_stats']['trust_level']})")
    print(f"{'='*70}")

    print(f"\n  金标来源: 人工={report['gold_source_stats']['human']}, "
          f"流水线={report['gold_source_stats']['pipeline_generated']}")

    print(f"\n  --- Layer 1: 文档级 ---")
    for k in [5, 10, 15]:
        key_r = f"doc_recall@{k}"
        key_p = f"doc_precision@{k}"
        if key_r in report["layer1"]:
            print(f"    Recall@{k}:    {report['layer1'][key_r]}")
            print(f"    Precision@{k}: {report['layer1'][key_p]}")
    print(f"    HitRate@10:  {report['layer1'].get('doc_hit_rate@10', 'N/A')}")

    print(f"\n  --- Layer 2: 片段级 ---")
    print(f"    Coverage@10:     {report['layer2'].get('snippet_coverage@10', 'N/A')}")
    print(f"    ChunkRecall@10:  {report['layer2'].get('chunk_recall@10', 'N/A')}")
    print(f"    ChunkPrecision@10: {report['layer2'].get('chunk_precision@10', 'N/A')}")

    print(f"\n  --- Layer 3: 排序质量 ---")
    print(f"    MRR@10:  {report['layer3'].get('mrr@10', 'N/A')}")
    print(f"    nDCG@10: {report['layer3'].get('ndcg@10', 'N/A')}")

    print(f"\n  --- Layer 4: 下游兼容 ---")
    print(f"    关键词完整度: {report['layer4'].get('keyword_completeness', 'N/A')}")

    if report.get("group_stats"):
        print(f"\n  --- 按问题类型 ---")
        for t, stats in report["group_stats"].items():
            print(f"    {t} (n={stats['n']}): recall={stats['recall@10']}, coverage={stats['coverage@10']}")

    if report.get("difficulty_stats"):
        print(f"\n  --- 按难度 ---")
        for d, stats in report["difficulty_stats"].items():
            print(f"    {d} (n={stats['n']}): recall={stats['recall@10']}, coverage={stats['coverage@10']}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于人工金标的评估")
    parser.add_argument("--dataset", type=str, default="eval_data/eval_dataset_human.json",
                        help="人工标注后的评估数据集路径")
    parser.add_argument("--mode", type=str, default="strict",
                        choices=["strict", "semi", "loose"],
                        help="评估模式: strict=仅人工金标, semi=全部但标记可信度, loose=全部等同")
    parser.add_argument("--force-rebuild", action="store_true", help="强制重建知识库")
    args = parser.parse_args()

    run_evaluation(args.dataset, args.mode, force_rebuild=args.force_rebuild)
