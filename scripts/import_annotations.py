"""
人工标注导入脚本
================
读取人工标注完成的 CSV，更新 data/eval/eval_dataset.json 中的金标。

核心逻辑：
  - 读取 CSV 中 is_relevant 列的人工标注结果
  - 只保留 is_relevant=1 的 chunk 作为新的金标
  - 将 gold_source 从 "pipeline_generated" 改为 "human"
  - 保留原始数据集的其他字段不变
  - 同时生成一份标注质量报告

用法：
  python scripts/import_annotations.py --csv data/eval/annotation_pool.csv
  python scripts/import_annotations.py --csv data/eval/annotation_pool.csv --output data/eval/eval_dataset_human.json
"""

import sys
import os
import json
import csv
import argparse
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def import_annotations(csv_path: str, dataset_path: str, output_path: str):
    """主流程：读取标注 CSV -> 更新数据集 -> 保存"""

    print("=" * 70)
    print("  人工标注导入 — 更新评估数据集金标")
    print("=" * 70)

    # 1. 读取标注 CSV
    print(f"\n[1/4] 读取标注文件: {csv_path}")
    annotations = defaultdict(list)  # query_id -> [chunk_id, ...]
    annotation_details = defaultdict(list)  # query_id -> [{chunk_id, relevance_level, notes}, ...]
    annotator_set = set()
    total_rows = 0
    annotated_rows = 0
    relevant_rows = 0
    not_relevant_rows = 0
    skipped_rows = 0

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            query_id = row.get("query_id", "")
            chunk_id = row.get("chunk_id", "")
            is_relevant = row.get("is_relevant", "").strip()
            relevance_level = row.get("relevance_level", "").strip()
            annotator = row.get("annotator", "").strip()
            notes = row.get("notes", "").strip()

            if not is_relevant:
                skipped_rows += 1
                continue

            if annotator:
                annotator_set.add(annotator)

            annotated_rows += 1

            if is_relevant == "1":
                relevant_rows += 1
                annotations[query_id].append(chunk_id)
                annotation_details[query_id].append({
                    "chunk_id": chunk_id,
                    "relevance_level": relevance_level,
                    "annotator": annotator,
                    "notes": notes,
                })
            elif is_relevant == "0":
                not_relevant_rows += 1
            else:
                print(f"  [警告] 无效标注值: query_id={query_id}, chunk_id={chunk_id}, is_relevant='{is_relevant}'")
                skipped_rows += 1

    print(f"  CSV 总行数:     {total_rows}")
    print(f"  已标注行数:     {annotated_rows}")
    print(f"  标记相关:       {relevant_rows}")
    print(f"  标记不相关:     {not_relevant_rows}")
    print(f"  未标注/跳过:    {skipped_rows}")
    print(f"  标注者:         {', '.join(annotator_set) if annotator_set else '未填写'}")

    if annotated_rows == 0:
        print("\n[错误] 没有找到任何标注结果！请先在 CSV 中填写 is_relevant 列。")
        return

    # 2. 加载原始数据集
    print(f"\n[2/4] 加载原始数据集: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    print(f"  原始题目数: {len(questions)}")

    # 3. 更新金标
    print(f"\n[3/4] 更新金标...")
    updated_count = 0
    unchanged_count = 0
    not_found_count = 0

    for q in questions:
        qid = q["query_id"]
        if qid in annotations:
            old_ids = set(q.get("relevant_chunk_ids", []))
            new_ids = annotations[qid]

            q["relevant_chunk_ids"] = new_ids
            q["gold_source"] = "human"
            q["annotation_date"] = datetime.now().strftime("%Y-%m-%d")
            q["annotation_details"] = annotation_details[qid]

            # 记录变更
            added = set(new_ids) - old_ids
            removed = old_ids - set(new_ids)
            q["annotation_diff"] = {
                "old_count": len(old_ids),
                "new_count": len(new_ids),
                "added": list(added),
                "removed": list(removed),
            }

            updated_count += 1
            print(f"  {qid}: {len(old_ids)} -> {len(new_ids)} 个金标 chunk "
                  f"(+{len(added)} -{len(removed)})")
        else:
            unchanged_count += 1
            # 保留原始 gold_source 但加警告
            if q.get("gold_source") == "pipeline_generated":
                q["needs_human_review"] = True

    not_found_count = len(set(annotations.keys()) - {q["query_id"] for q in questions})

    # 4. 保存新数据集
    print(f"\n[4/4] 保存新数据集: {output_path}")
    data["gold_source_stats"] = {
        "human": updated_count,
        "pipeline_generated": unchanged_count,
        "total": len(questions),
    }
    data["last_annotation_import"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 5. 生成标注质量报告
    report_path = output_path.replace(".json", "_annotation_report.json")
    report = {
        "import_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_csv": csv_path,
        "source_dataset": dataset_path,
        "output_dataset": output_path,
        "stats": {
            "total_csv_rows": total_rows,
            "annotated_rows": annotated_rows,
            "relevant_rows": relevant_rows,
            "not_relevant_rows": not_relevant_rows,
            "skipped_rows": skipped_rows,
            "annotators": list(annotator_set),
        },
        "dataset_update": {
            "updated_questions": updated_count,
            "unchanged_questions": unchanged_count,
            "csv_queries_not_in_dataset": not_found_count,
        },
        "per_query_summary": [],
    }

    for q in questions:
        qid = q["query_id"]
        diff = q.get("annotation_diff", {})
        report["per_query_summary"].append({
            "query_id": qid,
            "query": q["query"],
            "gold_source": q.get("gold_source", "unknown"),
            "old_gold_count": diff.get("old_count", len(q.get("relevant_chunk_ids", []))),
            "new_gold_count": diff.get("new_count", len(q.get("relevant_chunk_ids", []))),
            "added": diff.get("added", []),
            "removed": diff.get("removed", []),
        })

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 6. 打印汇总
    print(f"\n{'='*70}")
    print(f"  人工标注导入完成")
    print(f"{'='*70}")
    print(f"  更新题目数:     {updated_count}")
    print(f"  未变更题目数:   {unchanged_count}")
    print(f"  CSV中不在数据集: {not_found_count}")
    print(f"")
    print(f"  新数据集: {output_path}")
    print(f"  标注报告: {report_path}")
    print(f"")
    print(f"  下一步: 运行评估")
    print(f"    python scripts/run_eval_with_human_gold.py --dataset {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人工标注导入 — 更新评估数据集金标")
    parser.add_argument("--csv", type=str, default="eval_data/annotation_pool.csv",
                        help="人工标注完成的 CSV 文件路径")
    parser.add_argument("--dataset", type=str, default="data/eval/eval_dataset.json",
                        help="原始评估数据集路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出数据集路径（默认: data/eval/eval_dataset_human.json）")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.dataset.replace(".json", "_human.json")

    import_annotations(args.csv, args.dataset, args.output)
