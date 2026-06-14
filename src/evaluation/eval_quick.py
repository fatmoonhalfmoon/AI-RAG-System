"""
RAGAS风格评估 — Quick模式
==========================
快速评估：读取检索结果 + AI评估结果，计算指标，生成报告。

评估流程：
  Step 1: python scripts/collect_retrieval_results.py  (收集检索结果)
  Step 2: AI评估 (由IDE中的AI助手完成，输出 ragas_evaluation.json)
  Step 3: python -m src.evaluation.eval_quick           (计算指标+报告)
"""
import sys
import os
import time
import json
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from src.evaluation.dataset import RAGASEvaluation
from src.evaluation.metrics import RAGASMetrics, GroupMetrics, FailureAnalysis
from src.evaluation.reporter import EvalReporter


def run_eval_quick(eval_path: str = None):
    print("=" * 70)
    print("  RAG 检索质量评估 — RAGAS风格 Quick模式")
    print("=" * 70)

    # 1. 加载AI评估结果
    print("\n[1/3] 加载AI评估结果...")
    ragas_eval = RAGASEvaluation(eval_path)
    if not ragas_eval.evaluations:
        print("\n[错误] 没有AI评估结果！")
        print("  请先运行: python scripts/collect_retrieval_results.py")
        print("  然后让AI助手评估检索结果，输出 data/eval/ragas_evaluation.json")
        sys.exit(1)

    evaluations = ragas_eval.evaluations

    # 2. 计算指标
    print(f"\n[2/3] 计算RAGAS风格指标 ({len(evaluations)} 个query)...")
    report = compute_ragas_report(evaluations)

    # 3. 生成报告
    print("\n[3/3] 生成报告...")
    reporter = EvalReporter()
    reporter.save_json(report, "eval_ragas_quick_report.json")
    console_report = reporter.generate_console_report(report, mode="quick")
    print("\n" + console_report)

    return report


def compute_ragas_report(evaluations: List[Dict]) -> Dict:
    report = {
        "version": "3.0",
        "method": "ragas_style",
        "mode": "quick",
        "total_questions": len(evaluations),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {},
        "group_stats": {},
        "difficulty_stats": {},
        "failure_modes": {},
        "overall_grade": "",
        "bottleneck": "",
        "suggestion": "",
    }

    # 核心指标
    report["metrics"]["context_precision@5"] = RAGASMetrics.context_precision(evaluations, 5)
    report["metrics"]["context_precision@10"] = RAGASMetrics.context_precision(evaluations, 10)
    report["metrics"]["context_precision@15"] = RAGASMetrics.context_precision(evaluations, 15)
    report["metrics"]["context_relevance"] = RAGASMetrics.context_relevance(evaluations)
    report["metrics"]["context_sufficiency"] = RAGASMetrics.context_sufficiency(evaluations)
    report["metrics"]["ranking_quality@10"] = RAGASMetrics.ranking_quality(evaluations, 10)
    report["metrics"]["context_utilization"] = RAGASMetrics.context_utilization(evaluations)
    report["metrics"]["hit_rate@10"] = RAGASMetrics.hit_rate_at_k(evaluations, 10)

    # 分组统计
    report["group_stats"] = GroupMetrics.by_type(evaluations)
    report["difficulty_stats"] = GroupMetrics.by_difficulty(evaluations)

    # 失败模式
    report["failure_modes"] = FailureAnalysis.analyze(evaluations)

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

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAGAS风格评估 Quick模式")
    parser.add_argument("--eval-path", type=str, default=None, help="AI评估结果路径")
    args = parser.parse_args()

    run_eval_quick(eval_path=args.eval_path)
