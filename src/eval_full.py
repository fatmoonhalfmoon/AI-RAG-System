import sys
import os
import time
import json
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from src.main import RAGPipeline
from src.eval.dataset import EvalDataset
from src.eval.metrics import Layer1Metrics, Layer2Metrics, Layer3Metrics, Layer4Metrics
from src.eval.built_in_ai_judge import built_in_ai_judge
from src.eval.reporter import EvalReporter


def run_eval_full(dataset_path: str = None, top_k_values: List[int] = None, force_rebuild: bool = False):
    if top_k_values is None:
        top_k_values = [3, 5, 8, 10, 15]

    print("=" * 70)
    print("  RAG 检索质量评估 — Full 模式")
    print("=" * 70)

    print("\n[1/6] 加载评估数据集...")
    dataset = EvalDataset(dataset_path)

    # 检查金标来源
    if hasattr(dataset, 'gold_source_stats'):
        stats = dataset.gold_source_stats
        pipeline_qs = stats.get("pipeline_generated", 0)
        if pipeline_qs > 0:
            print(f"\n{'='*60}")
            print(f"  [重要警告] {pipeline_qs}/{stats['total']} 题的金标非人工标注")
            print(f"  评估指标可能被高估（信息泄漏问题）")
            print(f"  请运行 Pooling 标注流程修正:")
            print(f"    python scripts/generate_pool_for_annotation.py")
            print(f"{'='*60}")

    print("\n[2/6] 初始化 RAG 系统...")
    pipeline = RAGPipeline()
    pipeline.build_knowledge_base(force_rebuild=force_rebuild)

    print(f"\n[3/6] 对 {len(dataset)} 题进行检索评估...")
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
            "relevant_chunk_ids": q.get("relevant_chunk_ids", []),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "retrieved_docs": list(dict.fromkeys(c["source_doc"] for c in result.get("retrieved_chunks", []))),
            "merged_context": result.get("merged_context", ""),
            "source_docs": result.get("source_docs", []),
            "qa_hits": result.get("qa_hits", []),
            "retrieval_time_ms": result.get("retrieval_time_ms", 0),
        })

    print("\n[4/6] 计算四层指标...")
    report = compute_metrics_full(all_results, top_k_values)

    print("\n[5/6] 鲁棒性测试...")
    robustness_results = run_robustness_test(dataset, pipeline, top_k_values)
    report["robustness"] = robustness_results

    print("\n[6/6] 生成报告...")
    reporter = EvalReporter()
    reporter.save_json(report, "eval_full_report.json")
    console_report = reporter.generate_console_report(report, mode="full")
    print("\n" + console_report)

    return report


def compute_metrics_full(results: List[Dict], top_k_values: List[int]) -> Dict:
    report = {
        "version": "2.0",
        "mode": "full",
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

    # Layer 1: Document-level recall & precision
    for k in top_k_values:
        recalls = [Layer1Metrics.doc_recall_at_k(r["retrieved_docs"], r["relevant_docs"], k) for r in results]
        precisions = [Layer1Metrics.doc_precision_at_k(r["retrieved_docs"], r["relevant_docs"], k) for r in results]
        report["layer1"][f"doc_recall@{k}"] = round(sum(recalls) / len(recalls), 4) if recalls else 0
        report["layer1"][f"doc_precision@{k}"] = round(sum(precisions) / len(precisions), 4) if precisions else 0
    report["layer1"]["doc_hit_rate@10"] = Layer1Metrics.doc_hit_rate_at_k(results, 10)

    # Layer 2: Snippet coverage & chunk metrics
    coverages = []
    chunk_recalls = []
    chunk_precisions = []
    for r in results:
        text = " ".join(c.get("parent_content", c.get("content", "")) for c in r["retrieved_chunks"])
        cov = Layer2Metrics.snippet_coverage(text, r["answer_snippets"])
        coverages.append(cov)
        
        if r.get("relevant_chunk_ids"):
            chunk_recalls.append(Layer2Metrics.chunk_recall_at_k(r["retrieved_chunks"], r["relevant_chunk_ids"], 10))
            chunk_precisions.append(Layer2Metrics.chunk_precision_at_k(r["retrieved_chunks"], r["relevant_chunk_ids"], 10))
    
    report["layer2"]["snippet_coverage@10"] = round(sum(coverages) / len(coverages), 4) if coverages else 0
    report["layer2"]["chunk_recall@10"] = round(sum(chunk_recalls) / len(chunk_recalls), 4) if chunk_recalls else 0
    report["layer2"]["chunk_precision@10"] = round(sum(chunk_precisions) / len(chunk_precisions), 4) if chunk_precisions else 0

    # Layer 3: Ranking quality
    for k in top_k_values:
        report["layer3"][f"mrr@{k}"] = round(Layer3Metrics.mrr_at_k(results, k), 4)
        report["layer3"][f"ndcg@{k}"] = round(Layer3Metrics.ndcg_at_k(results, k), 4)

    # Layer 4: Downstream compatibility
    kw_completeness = []
    for r in results:
        kw = Layer4Metrics.keyword_completeness(r["merged_context"], r["reference_answer"])
        kw_completeness.append(kw)
    report["layer4"]["keyword_completeness"] = round(sum(kw_completeness) / len(kw_completeness), 4) if kw_completeness else 0

    fmt_checks = [Layer4Metrics.format_check(r) for r in results]
    fmt_pass = {k: all(c[k] for c in fmt_checks) for k in fmt_checks[0].keys()} if fmt_checks else {}
    report["layer4"]["format_check"] = fmt_pass

    # Group stats by query_type
    from collections import defaultdict
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

    # Failure mode analysis
    report["failure_modes"] = analyze_failure_modes(results)

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

    # Bottleneck & suggestion
    weak_groups = [t for t, s in report["group_stats"].items() if s["recall@10"] < 0.70]
    if weak_groups:
        report["bottleneck"] = "、".join(weak_groups) + " 类问题召回不足"
        report["suggestion"] = "增大 Top-K 到 15 + 增强查询扩展 + 优化向量模型"

    return report


def analyze_failure_modes(results: List[Dict]) -> Dict[str, int]:
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
    
    return failures


def run_robustness_test(dataset: EvalDataset, pipeline: RAGPipeline, top_k_values: List[int]) -> Dict:
    robustness_results = {"total_variants": 0, "recall_drop": 0, "details": []}
    
    for q in dataset.get_robustness_queries():
        variants = q.get("robustness_variants", [])
        base_result = pipeline.search(q["query"], top_k=10)
        base_recall = Layer1Metrics.doc_recall_at_k(
            list(dict.fromkeys(c["source_doc"] for c in base_result.get("retrieved_chunks", []))),
            q["relevant_docs"], 10
        )
        
        for variant in variants:
            variant_result = pipeline.search(variant, top_k=10)
            variant_recall = Layer1Metrics.doc_recall_at_k(
                list(dict.fromkeys(c["source_doc"] for c in variant_result.get("retrieved_chunks", []))),
                q["relevant_docs"], 10
            )
            drop = base_recall - variant_recall
            robustness_results["total_variants"] += 1
            robustness_results["recall_drop"] += drop
            robustness_results["details"].append({
                "query_id": q["query_id"],
                "variant": variant,
                "base_recall": base_recall,
                "variant_recall": variant_recall,
                "drop": drop,
            })
    
    if robustness_results["total_variants"] > 0:
        robustness_results["avg_recall_drop"] = round(
            robustness_results["recall_drop"] / robustness_results["total_variants"], 4
        )
    else:
        robustness_results["avg_recall_drop"] = 0
    
    return robustness_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="评估数据集路径")
    parser.add_argument("--force-rebuild", action="store_true", help="强制重建知识库")
    args = parser.parse_args()

    run_eval_full(dataset_path=args.dataset, force_rebuild=args.force_rebuild)
