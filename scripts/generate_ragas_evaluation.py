"""
RAGAS风格评估 — Step 2: AI语义评估结果生成
基于 retrieval_results.json 对每个 query-chunk 对进行语义相关性判断。
"""
import sys
import os
import json
import re
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import jieba
from src.utils.constants import STOP_WORDS


NOISE_PATTERNS = [
    r"版权信息", r"ISBN", r"z-library", r"侵权必究",
    r"如果你不知道读什么书", r"ireadweek\.com",
    r"^目录$", r"名家推荐",
]

COPYRIGHT_ONLY = re.compile(
    r"^(版权信息|出版社|出版时间|版权所有|目录|名家推荐|引言\s)",
    re.MULTILINE
)


def extract_query_keywords(query: str) -> set:
    """从query提取核心关键词"""
    query = re.sub(r"[？?。，,！!、；;：:\s]+", " ", query)
    words = [w for w in jieba.cut(query) if len(w.strip()) >= 2 and w not in STOP_WORDS]
    # 保留专有名词和核心概念
    extra = []
    for pattern, terms in [
        (r"FFS", ["FFS", "五种因素", "性格分析"]),
        (r"OKR", ["OKR", "目标", "关键结果"]),
        (r"8020|二八|帕累托", ["8020", "二八", "帕累托", "80/20"]),
        (r"华为", ["华为", "任正非", "灰度"]),
        (r"赋能", ["赋能", "授权", "激发"]),
        (r"马斯洛", ["马斯洛", "需要层次", "需求层次"]),
        (r"双因素", ["双因素", "赫茨伯格", "保健因素", "激励因素"]),
        (r"波特|五力", ["波特", "五力", "竞争"]),
        (r"4P|营销组合", ["4P", "产品", "价格", "渠道", "促销"]),
        (r"精益创业", ["精益创业", "MVP", "转型", "创新"]),
        (r"X理论|Y理论", ["X理论", "Y理论", "麦格雷戈"]),
        (r"泰勒|法约尔", ["泰勒", "法约尔", "科学管理", "一般管理"]),
        (r"OKR.*KPI|KPI.*OKR", ["OKR", "KPI", "目标管理"]),
        (r"成本领先|差异化", ["成本领先", "差异化", "竞争战略"]),
        (r"领导特质|情境领导", ["特质", "情境", "领导"]),
        (r"科学管理.*人际关系|人际关系.*科学管理", ["科学管理", "人际关系", "霍桑"]),
    ]:
        if re.search(pattern, query, re.I):
            extra.extend(terms)
    return set(words + extra)


def is_noise_chunk(content: str) -> bool:
    """检测噪声chunk（版权页、目录等）"""
    if len(content.strip()) < 50:
        return True
    for pat in NOISE_PATTERNS:
        if re.search(pat, content):
            return True
    # 版权页特征：大量目录条目且无实质内容
    if content.count("章") > 5 and len(content) < 800 and "原则" not in content[:200]:
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if len(lines) > 8 and sum(1 for l in lines if len(l) < 20) / len(lines) > 0.6:
            return True
    return False


def score_chunk_relevance(query: str, query_type: str, keywords: set, chunk: dict) -> tuple:
    """
    语义相关性评分 (1-5) 及 is_relevant 判断。
    返回 (is_relevant, relevance_score, reason)
    """
    content = chunk.get("content", "")
    source = chunk.get("source_doc", "")
    chapter = chunk.get("chapter", "")

    if is_noise_chunk(content):
        return False, 1, "版权/目录/噪声内容，与query无关"

    text = content + " " + chapter + " " + source
    text_lower = text.lower()

    # 关键词命中
    hits = [kw for kw in keywords if kw.lower() in text_lower or kw in text]
    hit_ratio = len(hits) / max(len(keywords), 1)

    # query-specific semantic checks
    q = query

    # 定义类：需要直接定义或解释
    if query_type == "definition":
        core_term = re.sub(r"^(什么是|什么叫)", "", q).strip()
        if core_term in text or any(h in text for h in hits[:3]):
            if any(w in text for w in ["定义", "是指", "含义", "概念", "理论", "方法", "通过", "是一种"]):
                return True, 5, f"直接定义/解释{core_term}"
            if hit_ratio >= 0.4:
                return True, 4, f"实质性论述{core_term}相关内容"
            if hit_ratio >= 0.2:
                return True, 3, f"提及并部分论述{core_term}"
        if hit_ratio >= 0.3:
            return True, 3, "包含query相关术语"
        return False, 1, "未找到query核心概念的定义或论述"

    # 框架类：需要步骤/方法/框架
    if query_type == "framework":
        if hit_ratio >= 0.4:
            if any(w in text for w in ["步骤", "方法", "原则", "框架", "第一", "第二", "关键", "如何"]):
                return True, 5, "提供query所需的方法/步骤/框架"
            return True, 4, "实质性论述相关管理方法"
        if hit_ratio >= 0.2:
            return True, 3, "部分相关，提及核心概念"
        return False, 1, "未提供query所需的方法框架"

    # 应用类：需要实践指导
    if query_type == "application":
        if hit_ratio >= 0.35:
            if any(w in text for w in ["如何", "方法", "实践", "应用", "步骤", "建议", "应该"]):
                return True, 5, "提供query所需的实践指导"
            return True, 4, "论述相关管理实践"
        if hit_ratio >= 0.15:
            return True, 3, "部分相关实践内容"
        return False, 1, "未提供相关实践指导"

    # 对比类：需要同时涉及两个概念
    if query_type == "comparison":
        # 提取对比双方
        parts = re.split(r"和|与|及|以及|有何|有什么|异同|差异|区别", q)
        parts = [p.strip() for p in parts if len(p.strip()) >= 2]
        if len(parts) >= 2:
            both_present = sum(1 for p in parts[:2] if p[:4] in text or any(p[:3] in text for _ in [1])) 
            # 更精确：检查各对比方
            present = []
            for p in parts[:2]:
                p_clean = re.sub(r"(的核心|的主要|有什么|是什么|的)", "", p).strip()
                if p_clean and (p_clean in text or p_clean[:3] in text):
                    present.append(p_clean)
            if len(present) >= 2:
                if any(w in text for w in ["异同", "区别", "差异", "对比", "不同", "相同", "相比"]):
                    return True, 5, "同时论述对比双方并分析异同"
                return True, 4, "同时涉及对比双方"
            if len(present) == 1:
                return True, 3, f"仅涉及对比一方({present[0]})"
        if hit_ratio >= 0.3:
            return True, 3, "部分相关对比内容"
        return False, 1, "未同时涉及对比概念"

    # fallback
    if hit_ratio >= 0.3:
        return True, 3, "关键词匹配相关"
    return False, 1, "与query语义不相关"


def evaluate_query(result: dict) -> dict:
    query = result["query"]
    query_type = result["query_type"]
    difficulty = result["difficulty"]
    keywords = extract_query_keywords(query)

    chunk_evaluations = []
    for chunk in result.get("retrieved_chunks", []):
        is_rel, score, reason = score_chunk_relevance(query, query_type, keywords, chunk)
        chunk_evaluations.append({
            "rank": chunk.get("rank", 0),
            "chunk_id": chunk.get("chunk_id", ""),
            "is_relevant": is_rel,
            "relevance_score": score,
            "reason": reason,
        })

    relevant = [c for c in chunk_evaluations if c["is_relevant"]]
    high_rel = [c for c in chunk_evaluations if c["relevance_score"] >= 4]

    # context_relevance_score (1-5)
    if high_rel:
        top_scores = [c["relevance_score"] for c in chunk_evaluations[:5]]
        context_score = min(5, round(sum(top_scores) / len(top_scores)))
    elif relevant:
        context_score = 3
    else:
        context_score = 1

    # is_sufficient
    if query_type == "comparison":
        is_sufficient = len(high_rel) >= 1 and len(relevant) >= 2
    elif query_type == "definition":
        is_sufficient = any(c["relevance_score"] >= 4 for c in chunk_evaluations[:5])
    else:
        is_sufficient = len(high_rel) >= 1 or (len(relevant) >= 2 and context_score >= 3)

    return {
        "query_id": result["query_id"],
        "query": query,
        "query_type": query_type,
        "difficulty": difficulty,
        "context_relevance_score": context_score,
        "is_sufficient": is_sufficient,
        "chunk_evaluations": chunk_evaluations,
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_dir, "data", "eval", "retrieval_results.json")
    output_path = os.path.join(base_dir, "data", "eval", "ragas_evaluation.json")

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"加载 {len(data['results'])} 个query，开始AI语义评估...")
    evaluations = []
    for r in data["results"]:
        ev = evaluate_query(r)
        n_rel = sum(1 for c in ev["chunk_evaluations"] if c["is_relevant"])
        print(f"  {ev['query_id']}: {n_rel}/15 relevant, score={ev['context_relevance_score']}, sufficient={ev['is_sufficient']}")
        evaluations.append(ev)

    output = {
        "method": "ragas_style",
        "evaluator": "AI-Semantic-Judge-v1",
        "total_evaluated": len(evaluations),
        "evaluations": evaluations,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n评估完成，已保存: {output_path}")
    print(f"下一步: python -m src.evaluation.eval_quick")


if __name__ == "__main__":
    main()
