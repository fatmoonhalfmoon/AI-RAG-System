"""
RAGAS风格评估 — 指标计算
==========================
基于AI评估结果计算RAGAS风格指标，无需金标。

核心指标：
  - Context Precision: 检索到的chunk中，有多少比例是相关的
  - Context Relevance: 检索到的上下文与query的相关程度
  - Context Sufficiency: 检索到的上下文是否足以回答query
  - Ranking Quality: 相关chunk是否排在前面
"""
import numpy as np
from typing import List, Dict


class RAGASMetrics:
    """RAGAS风格指标计算，基于AI评估结果"""

    @staticmethod
    def context_precision(evaluations: List[Dict], k: int = None) -> float:
        """Context Precision@K: 前K个检索chunk中相关chunk的比例

        基于AI对每个chunk的is_relevant判断。
        """
        precisions = []
        for e in evaluations:
            chunks = e.get("chunk_evaluations", [])
            if k is not None:
                chunks = chunks[:k]
            if not chunks:
                continue
            relevant_count = sum(1 for c in chunks if c.get("is_relevant", False))
            precisions.append(relevant_count / len(chunks))
        return round(sum(precisions) / len(precisions), 4) if precisions else 0.0

    @staticmethod
    def context_relevance(evaluations: List[Dict]) -> float:
        """Context Relevance: 检索上下文与query的整体相关度（1-5分归一化到0-1）"""
        scores = []
        for e in evaluations:
            score = e.get("context_relevance_score", 0)
            scores.append(score / 5.0)
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    @staticmethod
    def context_sufficiency(evaluations: List[Dict]) -> float:
        """Context Sufficiency: 检索上下文是否足以回答query的比例"""
        sufficient_count = sum(1 for e in evaluations if e.get("is_sufficient", False))
        return round(sufficient_count / len(evaluations), 4) if evaluations else 0.0

    @staticmethod
    def ranking_quality(evaluations: List[Dict], k: int = 10) -> float:
        """Ranking Quality: 相关chunk是否排在前面（类似MRR）

        对每个query，找到第一个相关chunk的排名位置，计算1/rank的均值。
        """
        rr_list = []
        for e in evaluations:
            chunks = e.get("chunk_evaluations", [])[:k]
            rr = 0.0
            for i, c in enumerate(chunks):
                if c.get("is_relevant", False):
                    rr = 1.0 / (i + 1)
                    break
            rr_list.append(rr)
        return round(sum(rr_list) / len(rr_list), 4) if rr_list else 0.0

    @staticmethod
    def context_utilization(evaluations: List[Dict]) -> float:
        """Context Utilization: 相关chunk在检索结果中的占比（衡量检索效率）"""
        utilizations = []
        for e in evaluations:
            chunks = e.get("chunk_evaluations", [])
            if not chunks:
                continue
            relevant = sum(1 for c in chunks if c.get("is_relevant", False))
            utilizations.append(relevant / len(chunks))
        return round(sum(utilizations) / len(utilizations), 4) if utilizations else 0.0

    @staticmethod
    def hit_rate_at_k(evaluations: List[Dict], k: int = 10) -> float:
        """Hit Rate@K: 前K个检索chunk中至少包含1个相关chunk的query比例

        衡量检索系统的基本召回能力：能否在Top-K中命中相关内容。
        """
        hits = 0
        for e in evaluations:
            chunks = e.get("chunk_evaluations", [])[:k]
            if any(c.get("is_relevant", False) for c in chunks):
                hits += 1
        return round(hits / len(evaluations), 4) if evaluations else 0.0


class GroupMetrics:
    """分组统计指标"""

    @staticmethod
    def by_type(evaluations: List[Dict]) -> Dict:
        from collections import defaultdict
        groups = defaultdict(list)
        for e in evaluations:
            groups[e.get("query_type", "unknown")].append(e)

        stats = {}
        for t, evals in groups.items():
            stats[t] = {
                "n": len(evals),
                "context_precision@10": RAGASMetrics.context_precision(evals, 10),
                "context_relevance": RAGASMetrics.context_relevance(evals),
                "context_sufficiency": RAGASMetrics.context_sufficiency(evals),
                "ranking_quality": RAGASMetrics.ranking_quality(evals, 10),
            }
        return stats

    @staticmethod
    def by_difficulty(evaluations: List[Dict]) -> Dict:
        from collections import defaultdict
        groups = defaultdict(list)
        for e in evaluations:
            groups[e.get("difficulty", "unknown")].append(e)

        stats = {}
        for d, evals in groups.items():
            stats[d] = {
                "n": len(evals),
                "context_precision@10": RAGASMetrics.context_precision(evals, 10),
                "context_relevance": RAGASMetrics.context_relevance(evals),
                "context_sufficiency": RAGASMetrics.context_sufficiency(evals),
            }
        return stats


class FailureAnalysis:
    """失败模式分析"""

    @staticmethod
    def analyze(evaluations: List[Dict]) -> Dict:
        failures = {
            "F1_zero_relevant": 0,      # 检索结果中没有任何相关chunk
            "F2_low_precision": 0,       # 精确率低于30%
            "F3_insufficient": 0,        # 上下文不足以回答query
            "F4_poor_ranking": 0,        # 相关chunk排在后面
            "F5_low_relevance": 0,       # 整体相关度评分低
        }

        for e in evaluations:
            chunks = e.get("chunk_evaluations", [])
            relevant_count = sum(1 for c in chunks if c.get("is_relevant", False))

            if relevant_count == 0:
                failures["F1_zero_relevant"] += 1
                continue

            precision = relevant_count / len(chunks) if chunks else 0
            if precision < 0.3:
                failures["F2_low_precision"] += 1

            if not e.get("is_sufficient", False):
                failures["F3_insufficient"] += 1

            # 检查第一个相关chunk的位置
            first_relevant_rank = None
            for i, c in enumerate(chunks):
                if c.get("is_relevant", False):
                    first_relevant_rank = i + 1
                    break
            if first_relevant_rank and first_relevant_rank > 5:
                failures["F4_poor_ranking"] += 1

            if e.get("context_relevance_score", 0) < 3:
                failures["F5_low_relevance"] += 1

        return failures
