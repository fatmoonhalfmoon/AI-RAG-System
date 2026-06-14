"""
RAGAS风格评估 — Full模式
==========================
完整评估：读取检索结果 + AI评估结果，计算详细指标，生成完整报告。
包含更细粒度的top-k分析和鲁棒性测试。

评估流程：
  Step 1: python scripts/collect_retrieval_results.py  (收集检索结果)
  Step 2: AI评估 (由IDE中的AI助手完成，输出 ragas_evaluation.json)
  Step 3: python -m src.evaluation.eval_full            (计算指标+报告)
"""
import sys
import os
import time
import json
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from src.evaluation.dataset import RetrievalResults, RAGASEvaluation
from src.evaluation.metrics import RAGASMetrics, GroupMetrics, FailureAnalysis
from src.evaluation.reporter import EvalReporter


def run_eval_full(eval_path: str = None, results_path: str = None):
    print("=" * 70)
    print("  RAG 检索质量评估 — RAGAS风格 Full模式")
    print("=" * 70)

    # 1. 加载AI评估结果
    print("\n[1/4] 加载AI评估结果...")
    ragas_eval = RAGASEvaluation(eval_path)
    if not ragas_eval.evaluations:
        print("\n[错误] 没有AI评估结果！")
        print("  请先运行: python scripts/collect_retrieval_results.py")
        print("  然后让AI助手评估检索结果，输出 data/eval/ragas_evaluation.json")
        sys.exit(1)

    evaluations = ragas_eval.evaluations

    # 2. 加载检索结果（用于鲁棒性测试等）
    print("\n[2/4] 加载检索结果...")
    try:
        retrieval = RetrievalResults(results_path)
    except Exception:
        retrieval = None
        print("  [警告] 未找到检索结果，跳过鲁棒性测试")

    # 3. 计算指标
    print(f"\n[3/4] 计算RAGAS风格指标 ({len(evaluations)} 个query)...")
    report = compute_ragas_report_full(evaluations)

    # 4. 生成报告
    print("\n[4/4] 生成报告...")
    reporter = EvalReporter()
    reporter.save_json(report, "eval_ragas_full_report.json")
    console_report = reporter.generate_console_report(report, mode="full")
    print("\n" + console_report)

    return report


def compute_ragas_report_full(evaluations: List[Dict]) -> Dict:
    report = {
        "version": "3.0",
        "method": "ragas_style",
        "mode": "full",
        "total_questions": len(evaluations),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {},
        "group_stats": {},
        "difficulty_stats": {},
        "failure_modes": {},
        "per_query_details": [],
        "overall_grade": "",
        "bottleneck": "",
        "suggestion": "",
    }

    # 核心指标 — 多个top-k值
    for k in [3, 5, 8, 10, 15]:
        report["metrics"][f"context_precision@{k}"] = RAGASMetrics.context_precision(evaluations, k)
        report["metrics"][f"ranking_quality@{k}"] = RAGASMetrics.ranking_quality(evaluations, k)
    report["metrics"][f"hit_rate@10"] = RAGASMetrics.hit_rate_at_k(evaluations, 10)

    # 整体指标
    report["metrics"]["context_relevance"] = RAGASMetrics.context_relevance(evaluations)
    report["metrics"]["context_sufficiency"] = RAGASMetrics.context_sufficiency(evaluations)
    report["metrics"]["context_utilization"] = RAGASMetrics.context_utilization(evaluations)

    # 分组统计
    report["group_stats"] = GroupMetrics.by_type(evaluations)
    report["difficulty_stats"] = GroupMetrics.by_difficulty(evaluations)

    # 失败模式
    report["failure_modes"] = FailureAnalysis.analyze(evaluations)

    # 每个query的详细结果
    for e in evaluations:
        chunks = e.get("chunk_evaluations", [])
        relevant_count = sum(1 for c in chunks if c.get("is_relevant", False))
        detail = {
            "query_id": e.get("query_id", ""),
            "query": e.get("query", ""),
            "query_type": e.get("query_type", ""),
            "difficulty": e.get("difficulty", ""),
            "total_chunks": len(chunks),
            "relevant_chunks": relevant_count,
            "precision": round(relevant_count / len(chunks), 4) if chunks else 0,
            "context_relevance_score": e.get("context_relevance_score", 0),
            "is_sufficient": e.get("is_sufficient", False),
            "first_relevant_rank": None,
        }
        for i, c in enumerate(chunks):
            if c.get("is_relevant", False):
                detail["first_relevant_rank"] = i + 1
                break
        report["per_query_details"].append(detail)

    # 总体评级
    precision = report["metrics"]["context_precision@10"]
    relevance = report["metrics"]["context_relevance"]
    sufficiency = report["metrics"]["context_sufficiency"]

    if precision >= 0.80 and relevance >= 0.80 and sufficiency >= 0.90:
        report["overall_grade"] = "A (优秀)"
    elif precision >= 0.65 and relevance >= 0.65 and sufficiency >= 0.75:
        report["overall_grade"] = "B+ (良好)"
    elif precision >= 0.50 and relevance >= 0.50 and sufficiency >= 0.60:
        report["overall_grade"] = "B (合格)"
    else:
        report["overall_grade"] = "C (需改进)"

    # 瓶颈分析
    weak_types = [t for t, s in report["group_stats"].items()
                  if s.get("context_precision@10", 1) < 0.50]
    if weak_types:
        report["bottleneck"] = "、".join(weak_types) + " 类问题检索精确率不足"
        report["suggestion"] = "优化查询扩展策略 + 增加文档级重排序 + 调整chunk切分粒度"

    fm = report["failure_modes"]
    if fm.get("F1_zero_relevant", 0) > len(evaluations) * 0.1:
        report["bottleneck"] = "部分query完全无法检索到相关内容"
        report["suggestion"] = "检查向量模型对专业术语的覆盖 + 增加BM25关键词召回通道"
    elif fm.get("F3_insufficient", 0) > len(evaluations) * 0.3:
        report["bottleneck"] = "检索上下文不足以回答query"
        report["suggestion"] = "增大Top-K + 优化chunk切分（增加上下文窗口）+ 增加父文档检索"

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAGAS风格评估 Full模式")
    parser.add_argument("--eval-path", type=str, default=None, help="AI评估结果路径")
    parser.add_argument("--results-path", type=str, default=None, help="检索结果路径")
    args = parser.parse_args()

    run_eval_full(eval_path=args.eval_path, results_path=args.results_path)
