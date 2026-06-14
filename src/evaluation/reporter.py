"""
RAGAS风格评估 — 报告生成器
============================
适配RAGAS风格指标的报告输出。
"""
import json
import os
from typing import Dict


class EvalReporter:
    """评估报告生成器"""

    def save_json(self, report: Dict, filename: str):
        """保存报告为 JSON"""
        output_dir = "data/output"
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def generate_console_report(self, report: Dict, mode: str = "ragas") -> str:
        """生成控制台可读的RAGAS风格报告"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"  RAG 检索质量评估报告 — RAGAS风格 ({mode.upper()} 模式)")
        lines.append("=" * 70)

        # 核心指标
        metrics = report.get("metrics", {})
        lines.append("\n  核心指标 (RAGAS-style)")
        lines.append("  " + "-" * 40)
        for key, label in [
            ("context_precision@5", "Context Precision@5"),
            ("context_precision@10", "Context Precision@10"),
            ("context_precision@15", "Context Precision@15"),
            ("context_relevance", "Context Relevance"),
            ("context_sufficiency", "Context Sufficiency"),
            ("ranking_quality@10", "Ranking Quality@10"),
            ("context_utilization", "Context Utilization"),
            ("hit_rate@10", "Hit Rate@10"),
        ]:
            if key in metrics:
                lines.append(f"    {label:30s} {metrics[key]}")

        # 分组统计
        if report.get("group_stats"):
            lines.append("\n  按问题类型统计")
            lines.append("  " + "-" * 40)
            for t, stats in report["group_stats"].items():
                lines.append(f"    {t} (n={stats['n']}):")
                lines.append(f"      Precision@10={stats.get('context_precision@10', 'N/A')}, "
                             f"Relevance={stats.get('context_relevance', 'N/A')}, "
                             f"Sufficiency={stats.get('context_sufficiency', 'N/A')}")

        if report.get("difficulty_stats"):
            lines.append("\n  按难度统计")
            lines.append("  " + "-" * 40)
            for d, stats in report["difficulty_stats"].items():
                lines.append(f"    {d} (n={stats['n']}): "
                             f"Precision@10={stats.get('context_precision@10', 'N/A')}, "
                             f"Sufficiency={stats.get('context_sufficiency', 'N/A')}")

        # 失败模式
        if report.get("failure_modes"):
            lines.append("\n  失败模式分析")
            lines.append("  " + "-" * 40)
            fm = report["failure_modes"]
            total = report.get("total_questions", 1)
            for key, desc in [
                ("F1_zero_relevant", "零相关chunk"),
                ("F2_low_precision", "低精确率(<30%)"),
                ("F3_insufficient", "上下文不足"),
                ("F4_poor_ranking", "排序靠后(>5)"),
                ("F5_low_relevance", "低相关度(<3分)"),
            ]:
                if key in fm:
                    lines.append(f"    {desc:20s} {fm[key]}/{total} ({fm[key]/total*100:.1f}%)")

        # 总体评级
        if report.get("overall_grade"):
            lines.append(f"\n  总体评级: {report['overall_grade']}")
        if report.get("bottleneck"):
            lines.append(f"  瓶颈: {report['bottleneck']}")
        if report.get("suggestion"):
            lines.append(f"  建议: {report['suggestion']}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)
