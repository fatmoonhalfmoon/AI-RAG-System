import json
import os
from typing import Dict, List
from datetime import datetime


class EvalReporter:
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(base_dir, "output")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_json(self, data: Dict, filename: str):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[报告] JSON 已保存: {path}")
        return path

    def generate_console_report(self, results: Dict, mode: str = "quick") -> str:
        lines = []
        mode_label = "Quick" if mode == "quick" else "Full"
        lines.append("╔" + "═" * 62 + "╗")
        lines.append("║" + " " * 12 + f"RAG 检索质量评估报告 — {mode_label} 模式" + " " * 13 + "║")
        lines.append("║" + f" {datetime.now().strftime('%Y-%m-%d')} | {results.get('total_questions', 0)} 题".ljust(61) + "║")
        lines.append("╠" + "═" * 62 + "╣")
        lines.append("║" + " " * 62 + "║")

        l1 = results.get("layer1", {})
        lines.append("║  Layer 1: 文档级召回".ljust(62) + "║")
        lines.append("║  " + "─" * 42 + " " * 18 + "║")
        for k in [5, 10, 15]:
            key = f"doc_recall@{k}"
            val = l1.get(key, 0)
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            lines.append(f"║  Doc Recall@{k}: {val:.2f} {bar}".ljust(62) + "║")
        hit_rate = l1.get("doc_hit_rate@10", 0)
        status = "通过" if hit_rate >= 0.90 else "警告"
        lines.append(f"║  Doc Hit Rate@10: {hit_rate:.2f} {status}".ljust(62) + "║")
        lines.append("║" + " " * 62 + "║")

        l2 = results.get("layer2", {})
        lines.append("║  Layer 2: 片段级覆盖".ljust(62) + "║")
        lines.append("║  " + "─" * 42 + " " * 18 + "║")
        cov = l2.get("snippet_coverage@10", 0)
        bar = "█" * int(cov * 20) + "░" * (20 - int(cov * 20))
        lines.append(f"║  Snippet Coverage@10: {cov:.2f} {bar}".ljust(62) + "║")
        if "chunk_recall@10" in l2:
            cr = l2["chunk_recall@10"]
            bar = "█" * int(cr * 20) + "░" * (20 - int(cr * 20))
            lines.append(f"║  Chunk Recall@10:     {cr:.2f} {bar}".ljust(62) + "║")
        lines.append("║" + " " * 62 + "║")

        l3 = results.get("layer3", {})
        lines.append("║  Layer 3: 排序质量".ljust(62) + "║")
        lines.append("║  " + "─" * 42 + " " * 18 + "║")
        mrr = l3.get("mrr@10", 0)
        bar = "█" * int(mrr * 20) + "░" * (20 - int(mrr * 20))
        lines.append(f"║  MRR@10:  {mrr:.2f} {bar}".ljust(62) + "║")
        ndcg = l3.get("ndcg@10", 0)
        bar = "█" * int(ndcg * 20) + "░" * (20 - int(ndcg * 20))
        lines.append(f"║  NDCG@10: {ndcg:.2f} {bar}".ljust(62) + "║")
        lines.append("║" + " " * 62 + "║")

        l4 = results.get("layer4", {})
        lines.append("║  Layer 4: 下游兼容性".ljust(62) + "║")
        lines.append("║  " + "─" * 42 + " " * 18 + "║")
        fmt = l4.get("format_check", {})
        fmt_pass = sum(1 for v in fmt.values() if v)
        lines.append(f"║  格式兼容性: 通过 ({fmt_pass}/{len(fmt)})".ljust(62) + "║")
        kw = l4.get("keyword_completeness", 0)
        bar = "█" * int(kw * 20) + "░" * (20 - int(kw * 20))
        lines.append(f"║  Keyword Completeness: {kw:.2f} {bar}".ljust(62) + "║")
        lines.append("║" + " " * 62 + "║")

        groups = results.get("group_stats", {})
        if groups:
            lines.append("║  分组统计".ljust(62) + "║")
            lines.append("║  " + "─" * 42 + " " * 18 + "║")
            lines.append("║  Type         | N  | Recall@10 | Coverage@10 | Flag".ljust(62) + "║")
            for t, stats in groups.items():
                flag = "通过" if stats.get("recall@10", 0) >= 0.85 else "警告"
                lines.append(f"║  {t:<12} | {stats.get('n', 0):>2} |   {stats.get('recall@10', 0):.2f}    |    {stats.get('coverage@10', 0):.2f}     | {flag}".ljust(62) + "║")
            lines.append("║" + " " * 62 + "║")

        diffs = results.get("difficulty_stats", {})
        if diffs:
            lines.append("║  难度统计".ljust(62) + "║")
            lines.append("║  " + "─" * 42 + " " * 18 + "║")
            lines.append("║  Difficulty  | N  | Recall@10 | Coverage@10".ljust(62) + "║")
            for d, stats in diffs.items():
                lines.append(f"║  {d:<12} | {stats.get('n', 0):>2} |   {stats.get('recall@10', 0):.2f}    |    {stats.get('coverage@10', 0):.2f}".ljust(62) + "║")
            lines.append("║" + " " * 62 + "║")

        failures = results.get("failure_modes", {})
        if failures:
            lines.append("║  失败模式分布".ljust(62) + "║")
            lines.append("║  " + "─" * 42 + " " * 18 + "║")
            for k, v in failures.items():
                lines.append(f"║  {k}: {v} 题".ljust(62) + "║")
            lines.append("║" + " " * 62 + "║")

        robustness = results.get("robustness", {})
        if robustness:
            lines.append("║  鲁棒性测试".ljust(62) + "║")
            lines.append("║  " + "─" * 42 + " " * 18 + "║")
            avg_drop = robustness.get("avg_recall_drop", 0)
            total_variants = robustness.get("total_variants", 0)
            lines.append(f"║  变体数: {total_variants} | 平均召回下降: {avg_drop:.2f}".ljust(62) + "║")
            lines.append("║" + " " * 62 + "║")

        overall = results.get("overall_grade", "N/A")
        bottleneck = results.get("bottleneck", "")
        suggestion = results.get("suggestion", "")
        lines.append(f"║  总体评级: {overall}".ljust(62) + "║")
        if bottleneck:
            lines.append(f"║  瓶颈: {bottleneck}".ljust(62) + "║")
        if suggestion:
            lines.append(f"║  建议: {suggestion}".ljust(62) + "║")
        lines.append("╚" + "═" * 62 + "╝")

        return "\n".join(lines)
