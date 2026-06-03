import json
import os
import re
from typing import Dict, List
from src.config import OUTPUT_DIR, MAX_MERGED_CONTEXT_LENGTH


class ResultFormatter:

    def _deduplicate_content(self, chunks: List[Dict]) -> List[Dict]:
        seen_signatures = set()
        result = []
        for chunk in chunks:
            content = chunk.get("parent_content", chunk.get("content", ""))
            signature = re.sub(r'\s+', '', content[:200])
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                result.append(chunk)
        return result

    def format_context(self, retrieval_result: Dict) -> str:
        chunks = retrieval_result.get("retrieved_chunks", [])
        qa_hits = retrieval_result.get("qa_hits", [])
        if not chunks and not qa_hits:
            return "未找到相关知识库内容。"

        deduped = self._deduplicate_content(chunks)

        lines = ["基于以下管理类知识库资料：\n"]
        source_docs = set()
        source_chunks: Dict[str, List[Dict]] = {}

        for chunk in deduped:
            source = chunk.get("source_doc", "未知来源")
            source_docs.add(source)
            if source not in source_chunks:
                source_chunks[source] = []
            source_chunks[source].append(chunk)

        for source in sorted(source_chunks.keys()):
            chunk_list = source_chunks[source]
            lines.append(f"【来源：{source}】")

            for chunk in chunk_list:
                content = chunk.get("parent_content", chunk.get("content", ""))
                if content and content.strip():
                    lines.append(content.strip())
                    lines.append("")

        if qa_hits:
            lines.append("【知识点匹配】")
            for qa in qa_hits:
                lines.append(f"问：{qa['question']}")
                lines.append(f"答：{qa['answer']}")
                lines.append(f"来源：{qa.get('source_doc', '')}")
                lines.append("")

        retrieval_result["source_docs"] = sorted(list(source_docs))
        result = "\n".join(lines)

        # 截断到最大长度限制，优先保留前面的内容（高rank chunk）
        if len(result) > MAX_MERGED_CONTEXT_LENGTH:
            # 预留截断提示文字的空间（约25字符），截断到 MAX_MERGED_CONTEXT_LENGTH - 25
            limit = MAX_MERGED_CONTEXT_LENGTH - 25
            result = result[:limit]
            # 避免截断在汉字中间，找到最后一个完整字符
            while len(result) > 0 and ord(result[-1]) > 127 and len(result.encode('utf-8')) % 3 != 0:
                result = result[:-1]
            result += "\n\n[内容已截断，以上为优先保留的相关片段]"

        return result

    def format_full_result(self, retrieval_result: Dict) -> str:
        chunks = retrieval_result.get("retrieved_chunks", [])
        qa_hits = retrieval_result.get("qa_hits", [])
        query = retrieval_result.get("query", "")
        merged_context = self.format_context(retrieval_result)
        sources = retrieval_result.get("source_docs", [])

        output = f"========== 检索结果 ==========\n"
        output += f"查询问题：{query}\n"
        output += f"查询扩展：{' | '.join(retrieval_result.get('expanded_queries', [query]))}\n"
        output += f"召回片段数：{len(chunks)} (去重后)\n"
        output += f"稠密检索候选：{retrieval_result.get('dense_candidates', 0)}\n"
        output += f"BM25检索候选：{retrieval_result.get('bm25_candidates', 0)}\n"
        output += f"Q&A库命中：{retrieval_result.get('qa_candidates', 0)}\n"
        output += f"RRF融合候选：{retrieval_result.get('fused_candidates', 0)}\n"
        output += f"来源去重候选：{retrieval_result.get('deduped_candidates', 0)}\n"
        output += f"检索耗时：{retrieval_result.get('retrieval_time_ms', 0):.1f}ms\n\n"
        output += "--- 片段详情 ---\n"

        for chunk in chunks:
            output += f"\n[{chunk['rank']}] 来源: {chunk['source_doc']} | "
            output += f"匹配类型: {chunk.get('match_type', 'N/A')} | "
            output += f"RRF={chunk['rrf_score']:.4f} | "
            output += f"Dense={chunk.get('dense_score', 0):.4f} | "
            output += f"BM25={chunk.get('bm25_score', 0):.4f}\n"
            merged_count = chunk.get("merged_count", 1)
            if merged_count > 1:
                output += f"(合并了{merged_count}个相邻片段)\n"
            chapter = chunk.get("chapter", "")
            if chapter:
                output += f"章节: {chapter}\n"
            content = chunk.get("content", "")
            output += f"内容: {content[:300]}{'...' if len(content) > 300 else ''}\n"

        if qa_hits:
            output += f"\n--- Q&A知识点命中 ---\n"
            for qa in qa_hits:
                output += f"\n问：{qa['question']}\n"
                output += f"答：{qa['answer'][:200]}{'...' if len(qa['answer']) > 200 else ''}\n"
                output += f"来源：{qa.get('source_doc', '')} | 类型：{qa.get('qa_type', '')} | 得分：{qa.get('score', 0):.4f}\n"

        output += f"\n--- 整合上下文（供下一步LLM使用） ---\n\n{merged_context}\n"
        output += f"\n--- 来源文档 ---\n"
        output += "\n".join(f"  - {s}" for s in sources)
        return output

    def save_result(self, retrieval_result: Dict, filename="retrieval_result"):
        filepath = os.path.join(OUTPUT_DIR, f"{filename}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(retrieval_result, f, ensure_ascii=False, indent=2)
        return filepath
