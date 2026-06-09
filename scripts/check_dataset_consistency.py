"""
数据一致性校验脚本
==================
检查 eval_dataset.json 中的金标数据是否一致、完整、可信。

检查项：
  1. gold_source 分布（pipeline_generated vs human）
  2. relevant_chunk_ids 是否在知识库索引中存在
  3. relevant_docs 是否与知识库文档名匹配
  4. 必填字段是否完整
  5. chunk_id 格式是否规范

用法：
  python scripts/check_dataset_consistency.py
  python scripts/check_dataset_consistency.py --dataset eval_data/eval_dataset_human.json
"""

import sys
import os
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def check_consistency(dataset_path: str):
    """主检查流程"""

    print("=" * 70)
    print("  评估数据集一致性校验")
    print("=" * 70)

    # 1. 加载数据集
    print(f"\n[1/3] 加载数据集: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    print(f"  题目数: {len(questions)}")

    # 2. 尝试加载知识库 chunk_ids（如果向量存储存在）
    known_chunk_ids = set()
    known_doc_names = set()

    vector_store_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_store")
    for size in ["small", "large"]:
        meta_path = os.path.join(vector_store_dir, size, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            for idx, meta in metadata.items():
                cid = meta.get("chunk_id", "")
                if cid:
                    known_chunk_ids.add(cid)
                doc = meta.get("source_doc", "")
                if doc:
                    known_doc_names.add(doc)

    print(f"  知识库 chunk 数: {len(known_chunk_ids)}")
    print(f"  知识库文档数: {len(known_doc_names)}")

    # 3. 逐项检查
    print(f"\n[2/3] 逐项检查...")
    issues = defaultdict(list)
    gold_source_dist = defaultdict(int)

    for q in questions:
        qid = q.get("query_id", "UNKNOWN")

        # 检查 1: 必填字段
        required_fields = ["query_id", "query", "query_type", "difficulty",
                           "relevant_docs", "reference_answer"]
        for field in required_fields:
            if field not in q or not q[field]:
                issues["missing_field"].append(f"{qid}.{field}")

        # 检查 2: gold_source
        source = q.get("gold_source", "unknown")
        gold_source_dist[source] += 1
        if source == "pipeline_generated":
            issues["pipeline_gold"].append(qid)
        elif source != "human":
            issues["unknown_gold_source"].append(f"{qid}={source}")

        # 检查 3: relevant_chunk_ids 存在性
        chunk_ids = q.get("relevant_chunk_ids", [])
        for cid in chunk_ids:
            if known_chunk_ids and cid not in known_chunk_ids:
                issues["missing_chunk"].append(f"{qid}.{cid}")

        # 检查 4: relevant_docs 匹配
        docs = q.get("relevant_docs", [])
        for doc in docs:
            if known_doc_names:
                matched = any(doc in known or known in doc for known in known_doc_names)
                if not matched:
                    issues["unmatched_doc"].append(f"{qid}.{doc}")

        # 检查 5: chunk_id 格式
        for cid in chunk_ids:
            parts = cid.split("_chunk_")
            if len(parts) != 2:
                issues["bad_chunk_format"].append(f"{qid}.{cid}")

    # 4. 汇总报告
    print(f"\n[3/3] 校验结果")
    print(f"\n  === 金标来源分布 ===")
    for source, count in sorted(gold_source_dist.items()):
        marker = " <-- 需人工审核" if source == "pipeline_generated" else ""
        print(f"    {source}: {count}{marker}")

    total_issues = sum(len(v) for v in issues.values())
    if total_issues == 0:
        print(f"\n  所有检查通过！数据集一致性良好。")
    else:
        print(f"\n  === 发现 {total_issues} 个问题 ===")
        for issue_type, items in sorted(issues.items()):
            print(f"\n  [{issue_type}] ({len(items)} 个)")
            for item in items[:10]:
                print(f"    - {item}")
            if len(items) > 10:
                print(f"    ... 还有 {len(items) - 10} 个")

    # 5. 保存报告
    report_path = dataset_path.replace(".json", "_consistency_report.json")
    report = {
        "dataset": dataset_path,
        "total_questions": len(questions),
        "gold_source_distribution": dict(gold_source_dist),
        "known_chunk_count": len(known_chunk_ids),
        "known_doc_count": len(known_doc_names),
        "total_issues": total_issues,
        "issues": {k: v for k, v in issues.items()},
        "trust_level": "high" if not issues.get("pipeline_gold") and total_issues == 0
                        else "medium" if not issues.get("pipeline_gold")
                        else "low",
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  校验报告已保存: {report_path}")

    if issues.get("pipeline_gold"):
        print(f"\n  [重要] {len(issues['pipeline_gold'])} 道题的金标为 pipeline_generated！")
        print(f"  请运行 Pooling 标注流程:")
        print(f"    python scripts/generate_pool_for_annotation.py")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估数据集一致性校验")
    parser.add_argument("--dataset", type=str, default="eval_data/eval_dataset.json",
                        help="评估数据集路径")
    args = parser.parse_args()

    check_consistency(args.dataset)
