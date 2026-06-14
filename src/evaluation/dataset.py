"""
RAGAS风格评估 — 检索结果加载器
================================
加载 collect_retrieval_results.py 生成的检索结果，
以及 AI 评估结果（ragas_evaluation.json）。

无需金标数据集，完全 reference-free。
"""


class RetrievalResults:
    """加载检索结果"""

    def __init__(self, results_path: str = None):
        import os
        import json
        from src.utils.path_utils import validate_path

        if results_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            results_path = os.path.join(base_dir, "data", "eval", "retrieval_results.json")
        self.results_path = results_path

        validate_path(self.results_path, must_exist=True)
        with open(self.results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.version = data.get("version", "1.0")
        self.total_queries = data.get("total_queries", 0)
        self.top_k = data.get("top_k", 15)
        self.system_config = data.get("system_config", {})
        self.results = data.get("results", [])

        print(f"[检索结果] 加载完成: {self.total_queries} 个query, top_k={self.top_k}")

    def get_by_type(self, query_type: str):
        return [r for r in self.results if r["query_type"] == query_type]

    def get_by_difficulty(self, difficulty: str):
        return [r for r in self.results if r["difficulty"] == difficulty]

    def get_by_query_id(self, query_id: str):
        for r in self.results:
            if r["query_id"] == query_id:
                return r
        return None

    def __len__(self):
        return len(self.results)

    def __iter__(self):
        return iter(self.results)


class RAGASEvaluation:
    """加载AI评估结果"""

    def __init__(self, eval_path: str = None):
        import os
        import json

        if eval_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            eval_path = os.path.join(base_dir, "data", "eval", "ragas_evaluation.json")
        self.eval_path = eval_path

        if not os.path.exists(eval_path):
            self.evaluations = []
            self.total_evaluated = 0
            print(f"[AI评估] 未找到评估结果: {eval_path}")
            print(f"[AI评估] 请先运行AI评估流程")
            return

        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.version = data.get("version", "1.0")
        self.total_evaluated = data.get("total_evaluated", 0)
        self.evaluations = data.get("evaluations", [])

        print(f"[AI评估] 加载完成: {self.total_evaluated} 个query已评估")

    def get_by_query_id(self, query_id: str):
        for e in self.evaluations:
            if e["query_id"] == query_id:
                return e
        return None

    def __len__(self):
        return len(self.evaluations)

    def __iter__(self):
        return iter(self.evaluations)
