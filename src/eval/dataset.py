import json
import os
from typing import List, Dict, Any


class EvalDataset:
    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            dataset_path = os.path.join(base_dir, "eval_data", "eval_dataset.json")
        self.dataset_path = dataset_path
        self.questions = []
        self.version = "1.0"
        self.total_questions = 0
        self._load()

    def _load(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"评估数据集不存在: {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.version = data.get("version", "1.0")
        self.total_questions = data.get("total_questions", 0)
        self.questions = data.get("questions", [])
        self._validate()

    def _validate(self):
        required_fields = ["query_id", "query", "query_type", "difficulty",
                           "relevant_docs", "answer_snippets", "reference_answer"]
        pipeline_count = 0
        human_count = 0
        other_count = 0
        for q in self.questions:
            for field in required_fields:
                if field not in q:
                    raise ValueError(f"题目 {q.get('query_id', '?')} 缺少必填字段: {field}")
            source = q.get("gold_source", "unknown")
            if source == "pipeline_generated":
                pipeline_count += 1
            elif source == "human":
                human_count += 1
            else:
                other_count += 1

        total = len(self.questions)
        print(f"[数据集] 加载完成: {total} 题, 版本 {self.version}")
        print(f"[数据集] 金标来源分布: 人工标注={human_count}, 流水线生成={pipeline_count}, 其他={other_count}")

        if pipeline_count > 0:
            print(f"[数据集] [警告] {pipeline_count}/{total} 题的金标为 pipeline_generated，评估结果可能被高估！")
            print(f"[数据集] [警告] 建议运行: python scripts/generate_pool_for_annotation.py")

        self.gold_source_stats = {
            "human": human_count,
            "pipeline_generated": pipeline_count,
            "other": other_count,
            "total": total,
            "trust_level": "high" if pipeline_count == 0 else "low",
        }

    def get_by_type(self, query_type: str) -> List[Dict]:
        return [q for q in self.questions if q["query_type"] == query_type]

    def get_by_difficulty(self, difficulty: str) -> List[Dict]:
        return [q for q in self.questions if q["difficulty"] == difficulty]

    def get_by_query_id(self, query_id: str) -> Dict:
        for q in self.questions:
            if q["query_id"] == query_id:
                return q
        return None

    def get_robustness_queries(self) -> List[Dict]:
        result = []
        for q in self.questions:
            if q.get("robustness_variants"):
                result.append(q)
        return result

    def __len__(self):
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)
