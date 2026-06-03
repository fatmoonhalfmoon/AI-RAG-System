import json
from typing import Dict


def built_in_ai_judge(query: str, merged_context: str, reference_answer: str) -> dict:
    """
    利用内置 AI 评估召回文本的充分性。
    无需调用外部 API，直接在当前环境中运行。
    支持多种 AI 模型（Kimi、GPT、Claude 等），通过统一接口调用。
    """
    prompt = f"""你是一个 RAG 系统检索质量评估专家。请判断以下检索结果是否足以支撑回答用户问题。

用户问题：{query}

检索到的上下文：
{merged_context}

标准参考答案：{reference_answer}

请从以下维度打分(1-5 分)：
1. 相关性(relevance)：检索文本与问题的相关程度
2. 完整性(completeness)：检索文本是否包含回答所需的全部关键信息
3. 准确性(accuracy)：检索文本中是否存在与问题矛盾或误导的内容
4. 简洁性(conciseness)：检索文本中无关内容的比例(5=无冗余)

输出 JSON：{{"relevance":X, "completeness":X, "accuracy":X, "conciseness":X, "can_answer":true/false, "missing_info":"..."}}"""

    try:
        # TODO: 接入实际AI调用接口
        # 当前为占位实现，返回基于启发式规则的评估
        # 实际部署时可通过环境变量或配置文件切换不同AI模型
        
        # 启发式评估：基于关键词匹配度给出粗略估计
        query_keywords = set(query.split())
        context_keywords = set(merged_context.split())
        overlap = len(query_keywords & context_keywords)
        relevance = min(5, max(1, int(overlap / max(1, len(query_keywords)) * 5)))
        
        ref_keywords = set(reference_answer.split())
        ref_overlap = len(ref_keywords & context_keywords)
        completeness = min(5, max(1, int(ref_overlap / max(1, len(ref_keywords)) * 5)))
        
        return {
            "relevance": relevance,
            "completeness": completeness,
            "accuracy": 4,
            "conciseness": 3,
            "can_answer": completeness >= 3,
            "missing_info": "" if completeness >= 3 else "检索内容可能不足以完整回答问题",
            "note": "这是启发式占位实现，建议接入实际AI模型以获得更准确评估"
        }
    except Exception as e:
        return {
            "relevance": 0,
            "completeness": 0,
            "accuracy": 0,
            "conciseness": 0,
            "can_answer": False,
            "missing_info": f"评估失败: {str(e)}",
            "note": "评估异常，请检查系统配置"
        }
