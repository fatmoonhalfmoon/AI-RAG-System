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
        for q in self.questions:
            for field in required_fields:
                if field not in q:
                    raise ValueError(f"题目 {q.get('query_id', '?')} 缺少必填字段: {field}")
        print(f"[数据集] 加载完成: {len(self.questions)} 题, 版本 {self.version}")

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
