import sys
import os
import time
import json
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from src.core.pipeline import RAGPipeline
from src.evaluation.dataset import EvalDataset
from src.evaluation.metrics import Layer1Metrics, Layer2Metrics, Layer3Metrics, Layer4Metrics
from src.evaluation.built_in_ai_judge import built_in_ai_judge
from src.evaluation.reporter import EvalReporter


def run_eval_quick(dataset_path: str = None, top_k_values: List[int] = None, force_rebuild: bool = False):
    if top_k_values is None:
        top_k_values = [5, 10, 15]

    print("=" * 70)
    print("  RAG 检索质量评估 — Quick 模式")
    print("=" * 70)
    from src.core.config import USE_CROSS_ENCODER, MMR_LAMBDA
    print("\n[当前配置]")
    print(f"  - Chunk重叠区: 100/200")
    print(f"  - 父文档窗口: 2000")
    print(f"  - RRF融合Top-K: 50/50/15")
    print(f"  - MMR权重: {MMR_LAMBDA} (动态调整)")
    print(f"  - 质量过滤: 8%")
    print(f"  - Cross-Encoder: {'已启用' if USE_CROSS_ENCODER else '暂不启用'}")
    print(f"  - Max Chunks/Doc: 5")
    print(f"  - 缓存目录: data/vector_store/")

    print("\n[1/5] 加载评估数据集...")
    dataset = EvalDataset(dataset_path)

    print("\n[2/5] 初始化 RAG 系统...")
    pipeline = RAGPipeline()
    pipeline.build_knowledge_base(force_rebuild=force_rebuild)

    print(f"\n[3/5] 对 {len(dataset)} 题进行检索评估...")
    all_results = []
    for q in dataset.questions:
        result = pipeline.search(q["query"], top_k=max(top_k_values))
        all_results.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "query_type": q["query_type"],
            "difficulty": q["difficulty"],
            "relevant_docs": q["relevant_docs"],
            "answer_snippets": q["answer_snippets"],
            "reference_answer": q["reference_answer"],
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "retrieved_docs": list(dict.fromkeys(c["source_doc"] for c in result.get("retrieved_chunks", []))),
            "merged_context": result.get("merged_context", ""),
            "source_docs": result.get("source_docs", []),
            "qa_hits": result.get("qa_hits", []),
            "retrieval_time_ms": result.get("retrieval_time_ms", 0),
        })

    print("\n[4/5] 计算四层指标...")
    report = compute_metrics(all_results, top_k_values)

    print("\n[5/5] 生成报告...")
    reporter = EvalReporter()
    reporter.save_json(report, "eval_quick_report.json")
    console_report = reporter.generate_console_report(report, mode="quick")
    print("\n" + console_report)

    return report


def compute_metrics(results: List[Dict], top_k_values: List[int]) -> Dict:
    report = {
        "version": "2.1",
        "mode": "quick",
        "total_questions": len(results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layer1": {},
        "layer2": {},
        "layer3": {},
        "layer4": {},
        "group_stats": {},
        "difficulty_stats": {},
        "failure_modes": {},
        "overall_grade": "",
        "bottleneck": "",
        "suggestion": "",
    }

    # Layer 1
    for k in top_k_values:
        recalls = [Layer1Metrics.doc_recall_at_k(r["retrieved_docs"], r["relevant_docs"], k) for r in results]
        report["layer1"][f"doc_recall@{k}"] = round(sum(recalls) / len(recalls), 4) if recalls else 0
    report["layer1"]["doc_hit_rate@10"] = Layer1Metrics.doc_hit_rate_at_k(results, 10)

    # Layer 2
    coverages = []
    for r in results:
        text = " ".join(c.get("parent_content", c.get("content", "")) for c in r["retrieved_chunks"])
        cov = Layer2Metrics.snippet_coverage(text, r["answer_snippets"])
        coverages.append(cov)
    report["layer2"]["snippet_coverage@10"] = round(sum(coverages) / len(coverages), 4) if coverages else 0

    # Layer 3
    report["layer3"]["mrr@10"] = round(Layer3Metrics.mrr_at_k(results, 10), 4)
    report["layer3"]["ndcg@10"] = round(Layer3Metrics.ndcg_at_k(results, 10), 4)

    # Layer 4
    kw_completeness = []
    for r in results:
        kw = Layer4Metrics.keyword_completeness(r["merged_context"], r["reference_answer"])
        kw_completeness.append(kw)
    report["layer4"]["keyword_completeness"] = round(sum(kw_completeness) / len(kw_completeness), 4) if kw_completeness else 0

    fmt_checks = [Layer4Metrics.format_check(r) for r in results]
    fmt_pass = {k: all(c[k] for c in fmt_checks) for k in fmt_checks[0].keys()} if fmt_checks else {}
    report["layer4"]["format_check"] = fmt_pass

    # Group stats
    from collections import defaultdict
    groups = defaultdict(lambda: {"n": 0, "recalls": [], "coverages": []})
    for r in results:
        t = r["query_type"]
        groups[t]["n"] += 1
        groups[t]["recalls"].append(Layer1Metrics.doc_recall_at_k(r["retrieved_docs"], r["relevant_docs"], 10))
        text = " ".join(c.get("parent_content", c.get("content", "")) for c in r["retrieved_chunks"])
        groups[t]["coverages"].append(Layer2Metrics.snippet_coverage(text, r["answer_snippets"]))

    for t, stats in groups.items():
        report["group_stats"][t] = {
            "n": stats["n"],
            "recall@10": round(sum(stats["recalls"]) / len(stats["recalls"]), 4),
            "coverage@10": round(sum(stats["coverages"]) / len(stats["coverages"]), 4),
        }

    # Difficulty stats
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

    # Failure modes
    failures = {"F1": 0, "F2": 0, "F3": 0, "F4": 0, "F5": 0, "F6": 0}
    for r in results:
        recall = Layer1Metrics.doc_recall_at_k(r["retrieved_docs"], r["relevant_docs"], 10)
        if recall == 0:
            failures["F1"] += 1
        elif recall < 0.5:
            failures["F2"] += 1
        
        text = " ".join(c.get("parent_content", c.get("content", "")) for c in r["retrieved_chunks"])
        coverage = Layer2Metrics.snippet_coverage(text, r["answer_snippets"])
        if coverage < 0.5:
            failures["F3"] += 1
        
        mrr = Layer3Metrics.mrr_at_k([r], 10)
        if mrr < 0.5 and recall > 0:
            failures["F4"] += 1
        
        kw = Layer4Metrics.keyword_completeness(r["merged_context"], r["reference_answer"])
        if kw < 0.5 and recall > 0:
            failures["F5"] += 1
        
        if recall > 0 and coverage < 0.3:
            failures["F6"] += 1
    
    report["failure_modes"] = failures

    # Overall grade
    l1_score = report["layer1"]["doc_recall@10"]
    l2_score = report["layer2"]["snippet_coverage@10"]
    if l1_score >= 0.95 and l2_score >= 0.85:
        report["overall_grade"] = "A (优秀)"
    elif l1_score >= 0.85 and l2_score >= 0.70:
        report["overall_grade"] = "B+ (良好)"
    elif l1_score >= 0.70 and l2_score >= 0.55:
        report["overall_grade"] = "B (合格)"
    else:
        report["overall_grade"] = "C (需改进)"

    # Bottleneck
    weak_groups = [t for t, s in report["group_stats"].items() if s["recall@10"] < 0.70]
    if weak_groups:
        report["bottleneck"] = "、".join(weak_groups) + " 类问题召回不足"
        report["suggestion"] = "增大 Top-K + 增强查询扩展 + 优化向量模型"

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="评估数据集路径")
    parser.add_argument("--force-rebuild", action="store_true", help="强制重建知识库")
    args = parser.parse_args()

    run_eval_quick(dataset_path=args.dataset, force_rebuild=args.force_rebuild)
