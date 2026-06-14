"""
RAGAS风格评估 — 检索结果收集器
================================
运行RAG系统对测试query进行检索，收集结果供AI评估。
无需金标数据集，完全reference-free。

评估流程：
  Step 1: 本脚本收集检索结果 → data/eval/retrieval_results.json
  Step 2: AI读取结果进行评估 → data/eval/ragas_evaluation.json
  Step 3: 计算指标生成报告
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from src.core.pipeline import RAGPipeline
from src.core.config import USE_CROSS_ENCODER, MMR_LAMBDA, CROSS_ENCODER_CACHE_ENABLED


# 测试query集合 — 覆盖知识库所有文档，4种类型
EVAL_QUERIES = [
    # ===== 定义类 (Definition) =====
    {"query_id": "Q001", "query": "什么是FFS理论", "query_type": "definition", "difficulty": "easy"},
    {"query_id": "Q002", "query": "什么是OKR工作法", "query_type": "definition", "difficulty": "easy"},
    {"query_id": "Q003", "query": "什么是产品思维", "query_type": "definition", "difficulty": "easy"},
    {"query_id": "Q004", "query": "什么是公共管理", "query_type": "definition", "difficulty": "medium"},
    {"query_id": "Q005", "query": "什么是赋能型领导", "query_type": "definition", "difficulty": "medium"},
    {"query_id": "Q006", "query": "什么是华为灰度管理", "query_type": "definition", "difficulty": "medium"},
    {"query_id": "Q007", "query": "什么是组织文化", "query_type": "definition", "difficulty": "easy"},
    {"query_id": "Q008", "query": "什么是马斯洛需求层次理论", "query_type": "definition", "difficulty": "easy"},
    {"query_id": "Q009", "query": "什么是双因素理论", "query_type": "definition", "difficulty": "easy"},
    {"query_id": "Q010", "query": "什么是企业的宗旨和使命", "query_type": "definition", "difficulty": "medium"},
    {"query_id": "Q011", "query": "什么是8020法则", "query_type": "definition", "difficulty": "easy"},
    {"query_id": "Q012", "query": "什么是波特五力模型", "query_type": "definition", "difficulty": "easy"},
    {"query_id": "Q013", "query": "什么是市场营销4P理论", "query_type": "definition", "difficulty": "easy"},
    {"query_id": "Q014", "query": "什么是精益创业", "query_type": "definition", "difficulty": "easy"},
    {"query_id": "Q015", "query": "什么是管理思维中的目标原则", "query_type": "definition", "difficulty": "easy"},
    # ===== 框架类 (Framework) =====
    {"query_id": "Q016", "query": "小团队主管的四项工作是什么", "query_type": "framework", "difficulty": "easy"},
    {"query_id": "Q017", "query": "如何通过倾听提升下属的工作动力", "query_type": "framework", "difficulty": "medium"},
    {"query_id": "Q018", "query": "如何进行有效的会议管理", "query_type": "framework", "difficulty": "easy"},
    {"query_id": "Q019", "query": "OKR工作法的实施步骤是什么", "query_type": "framework", "difficulty": "easy"},
    {"query_id": "Q020", "query": "用户体验的五个层次是什么", "query_type": "framework", "difficulty": "medium"},
    {"query_id": "Q021", "query": "点线面体战略选择的核心逻辑是什么", "query_type": "framework", "difficulty": "medium"},
    {"query_id": "Q022", "query": "新手管理者的三个关键跨越是什么", "query_type": "framework", "difficulty": "easy"},
    {"query_id": "Q023", "query": "辅导员工的关键四步是什么", "query_type": "framework", "difficulty": "medium"},
    {"query_id": "Q024", "query": "华为供应链管理的深淘滩低作堰原则是什么", "query_type": "framework", "difficulty": "medium"},
    {"query_id": "Q025", "query": "德鲁克认为管理者如何掌握自己的时间", "query_type": "framework", "difficulty": "medium"},
    {"query_id": "Q026", "query": "卓有成效的管理者如何发挥人的长处", "query_type": "framework", "difficulty": "medium"},
    {"query_id": "Q027", "query": "基业长青的公司有什么共同特质", "query_type": "framework", "difficulty": "medium"},
    {"query_id": "Q028", "query": "如何进行市场细分", "query_type": "framework", "difficulty": "easy"},
    {"query_id": "Q029", "query": "如何进行市场定位", "query_type": "framework", "difficulty": "easy"},
    {"query_id": "Q030", "query": "竞争战略的基本类型有哪些", "query_type": "framework", "difficulty": "easy"},
    {"query_id": "Q031", "query": "如何打造团队执行力", "query_type": "framework", "difficulty": "medium"},
    {"query_id": "Q032", "query": "如何进行组织变革管理", "query_type": "framework", "difficulty": "medium"},
    {"query_id": "Q033", "query": "如何使工作富有成效", "query_type": "framework", "difficulty": "medium"},
    {"query_id": "Q034", "query": "如何管理知识工作者", "query_type": "framework", "difficulty": "medium"},
    {"query_id": "Q035", "query": "创新核算制度的关键是什么", "query_type": "framework", "difficulty": "medium"},
    # ===== 应用类 (Application) =====
    {"query_id": "Q036", "query": "小团队如何有效管理", "query_type": "application", "difficulty": "easy"},
    {"query_id": "Q037", "query": "如何成为卓有成效的管理者", "query_type": "application", "difficulty": "medium"},
    {"query_id": "Q038", "query": "华为如何管理危机意识", "query_type": "application", "difficulty": "medium"},
    {"query_id": "Q039", "query": "如何应用精益创业方法改进产品", "query_type": "application", "difficulty": "medium"},
    {"query_id": "Q040", "query": "如何从业务高手转变为优秀主管", "query_type": "application", "difficulty": "medium"},
    {"query_id": "Q041", "query": "如何提供优良的顾客价值", "query_type": "application", "difficulty": "easy"},
    {"query_id": "Q042", "query": "如何应用8020法则提升管理效率", "query_type": "application", "difficulty": "medium"},
    {"query_id": "Q043", "query": "如何进行有效的绩效反馈", "query_type": "application", "difficulty": "medium"},
    # ===== 对比类 (Comparison) =====
    {"query_id": "Q044", "query": "OKR和KPI有什么异同", "query_type": "comparison", "difficulty": "medium"},
    {"query_id": "Q045", "query": "泰勒和法约尔的管理思想有何异同", "query_type": "comparison", "difficulty": "hard"},
    {"query_id": "Q046", "query": "X理论和Y理论的核心差异是什么", "query_type": "comparison", "difficulty": "easy"},
    {"query_id": "Q047", "query": "领导特质理论和情境领导理论有何差异", "query_type": "comparison", "difficulty": "medium"},
    {"query_id": "Q048", "query": "科学管理理论和人际关系学派的主要区别是什么", "query_type": "comparison", "difficulty": "medium"},
    {"query_id": "Q049", "query": "成本领先战略和差异化战略的适用条件有何不同", "query_type": "comparison", "difficulty": "medium"},
    {"query_id": "Q050", "query": "马斯洛需求层次论和双因素理论的异同", "query_type": "comparison", "difficulty": "medium"},
]


def collect_retrieval_results(output_path: str, top_k: int = 15, force_rebuild: bool = False):
    """运行RAG系统收集检索结果"""

    print("=" * 70)
    print("  RAGAS风格评估 — 检索结果收集")
    print("=" * 70)

    # 1. 初始化RAG系统
    print(f"\n[1/3] 初始化RAG系统...")
    pipeline = RAGPipeline()
    pipeline.build_knowledge_base(force_rebuild=force_rebuild)

    # 2. 对每个query进行检索
    print(f"\n[2/3] 对 {len(EVAL_QUERIES)} 个query进行检索 (top_k={top_k})...")
    results = []

    for i, q in enumerate(EVAL_QUERIES):
        query_id = q["query_id"]
        query = q["query"]

        start_time = time.time()
        search_result = pipeline.search(query, top_k=top_k)
        elapsed = (time.time() - start_time) * 1000

        # 提取检索到的chunk信息
        retrieved_chunks = []
        for chunk in search_result.get("retrieved_chunks", []):
            chunk_entry = {
                "rank": chunk.get("rank", 0),
                "chunk_id": chunk.get("chunk_id", ""),
                "source_doc": chunk.get("source_doc", ""),
                "chapter": chunk.get("chapter", ""),
                "content": chunk.get("content", ""),
                "rrf_score": chunk.get("rrf_score", 0),
                "match_type": chunk.get("match_type", ""),
            }
            if chunk.get("cross_encoder_score") is not None:
                chunk_entry["cross_encoder_score"] = round(chunk["cross_encoder_score"], 4)
            retrieved_chunks.append(chunk_entry)

        result = {
            "query_id": query_id,
            "query": query,
            "query_type": q["query_type"],
            "difficulty": q["difficulty"],
            "retrieved_chunks": retrieved_chunks,
            "source_docs": search_result.get("source_docs", []),
            "merged_context": search_result.get("merged_context", ""),
            "retrieval_time_ms": round(elapsed, 1),
            "qa_hits": search_result.get("qa_hits", []),
        }
        results.append(result)

        # 进度
        n_chunks = len(retrieved_chunks)
        n_docs = len(set(c["source_doc"] for c in retrieved_chunks))
        print(f"  [{i+1}/{len(EVAL_QUERIES)}] {query_id}: {n_chunks} chunks, {n_docs} docs, {elapsed:.0f}ms")

    # 3. 保存结果
    print(f"\n[3/3] 保存检索结果到: {output_path}")
    output = {
        "version": "1.0",
        "method": "ragas_style",
        "created": time.strftime("%Y-%m-%d"),
        "total_queries": len(results),
        "top_k": top_k,
        "system_config": {
            "use_cross_encoder": USE_CROSS_ENCODER,
            "cross_encoder_cache_enabled": CROSS_ENCODER_CACHE_ENABLED,
            "mmr_lambda": MMR_LAMBDA,
        },
        "results": results,
    }
    if pipeline.retriever and pipeline.retriever._cross_encoder:
        output["cross_encoder_stats"] = pipeline.retriever._cross_encoder.get_stats()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 统计
    total_chunks = sum(len(r["retrieved_chunks"]) for r in results)
    avg_time = sum(r["retrieval_time_ms"] for r in results) / len(results)
    print(f"\n{'='*70}")
    print(f"  检索结果收集完成")
    print(f"{'='*70}")
    print(f"  总query数: {len(results)}")
    print(f"  总检索chunk数: {total_chunks}")
    print(f"  平均每query: {total_chunks/len(results):.1f} chunks")
    print(f"  平均检索耗时: {avg_time:.0f}ms")
    print(f"\n  下一步: 请让AI读取 {output_path} 进行RAGAS评估")
    print(f"{'='*70}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS风格评估 — 检索结果收集")
    parser.add_argument("--output", type=str, default="data/eval/retrieval_results.json",
                        help="检索结果输出路径")
    parser.add_argument("--top-k", type=int, default=15, help="检索top-k（默认15）")
    parser.add_argument("--force-rebuild", action="store_true", help="强制重建索引")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, args.output) if not os.path.isabs(args.output) else args.output

    collect_retrieval_results(output_path, top_k=args.top_k, force_rebuild=args.force_rebuild)
