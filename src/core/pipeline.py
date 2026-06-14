import os
import sys
import time
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from src.core.config import (
    KNOWLEDGE_BASE_DIR, VECTOR_STORE_DIR, OUTPUT_DIR,
    CHUNK_SIZE_SMALL, CHUNK_SIZE_LARGE, PARENT_CHUNK_SIZE,
    FINAL_TOP_K, DENSE_TOP_K, BM25_TOP_K, QA_TOP_K,
    DENSE_MAX_CHUNKS, VECTOR_MIN_PER_DOC,
    TEST_QUESTIONS,
)
from src.indexing.text_splitter import (
    ChineseTextSplitter, load_knowledge_base, select_dense_chunks, ChunkQualityFilter,
)
from src.indexing.embedding import EmbeddingModel
from src.indexing.vector_store import FAISSVectorStore
from src.indexing.qa_extractor import QAExtractor, QARetriever
from src.retrieval.retriever import HybridRetriever
from src.retrieval.formatter import ResultFormatter


class RAGPipeline:
    def __init__(self):
        self.small_splitter = ChineseTextSplitter.create_small_splitter()
        self.large_splitter = ChineseTextSplitter.create_large_splitter()
        self.embedding_model = None
        self.small_vector_store = None
        self.large_vector_store = None
        self.retriever = None
        self.formatter = ResultFormatter()
        self.qa_extractor = QAExtractor()
        self.qa_retriever = None
        self.all_chunks = []
        self.small_dense_embeddings = None
        self.large_dense_embeddings = None
        self.qa_pairs = []

    def build_knowledge_base(self, kb_dir=KNOWLEDGE_BASE_DIR, force_rebuild=False):
        print("=" * 60)
        print("  七阶段精炼管道 — 构建管理类知识库")
        print("=" * 60)

        if not force_rebuild and self._check_cache():
            print("[跳过] 向量存储已存在，加载缓存...")
            self._load_from_cache()
            return

        print(f"\n--- Stage 1: 文档名规范化 + 繁简统一 + 领域词典注入 ---")
        print(f"读取知识库文档: {kb_dir}")
        documents = load_knowledge_base(kb_dir)
        if not documents:
            raise ValueError(f"知识库目录为空或不存在: {kb_dir}")
        print(f"共加载 {len(documents)} 份文档")
        total_refined = sum(len(d["content"]) for d in documents)
        print(f"精炼后总文本量: {total_refined // 1000}K 字符")

        print(f"\n--- Stage 2: 双粒度切分 + chunk级质量过滤 ---")
        all_small_chunks = []
        all_large_chunks = []

        for doc in documents:
            small_chunks = self.small_splitter.create_chunks_with_parents(doc["content"], doc["doc_name"])
            large_chunks = self.large_splitter.create_chunks_with_parents(doc["content"], doc["doc_name"])
            all_small_chunks.extend(small_chunks)
            all_large_chunks.extend(large_chunks)
            print(f"  {doc['doc_name'][:30]}: 小粒度{len(small_chunks)}个, 大粒度{len(large_chunks)}个")

        print(f"\n小粒度chunk数: {len(all_small_chunks)}")
        print(f"大粒度chunk数: {len(all_large_chunks)}")

        quality_filter = ChunkQualityFilter()
        filtered_small = quality_filter.filter_chunks(all_small_chunks)
        print(f"小粒度质量过滤后: {len(filtered_small)}")
        filtered_large = quality_filter.filter_chunks(all_large_chunks)
        print(f"大粒度质量过滤后: {len(filtered_large)}")

        all_chunks = filtered_small + filtered_large
        self.all_chunks = all_chunks

        dense_small = select_dense_chunks(filtered_small, DENSE_MAX_CHUNKS)
        print(f"小粒度Dense采样: {len(dense_small)}")
        dense_large = select_dense_chunks(filtered_large, max(DENSE_MAX_CHUNKS // 2, VECTOR_MIN_PER_DOC * len(documents)))
        print(f"大粒度Dense采样: {len(dense_large)}")

        print(f"\n--- Stage 3a: 语义向量化(BGE-large-zh-v1.5) + 向量质量清洗 ---")
        self.embedding_model = EmbeddingModel()

        print(f"\n[小粒度向量化]")
        small_texts = [c["content"] for c in dense_small]
        small_embeddings = self.embedding_model.encode_documents(small_texts)
        print(f"原始小粒度向量数: {len(small_embeddings)}, 维度: {small_embeddings.shape[1]}")

        temp_store_small = FAISSVectorStore(self.embedding_model.dim)
        small_embeddings, dense_small = temp_store_small.clean_embeddings(small_embeddings, dense_small)

        print(f"\n[大粒度向量化]")
        large_texts = [c["content"] for c in dense_large]
        large_embeddings = self.embedding_model.encode_documents(large_texts)
        print(f"原始大粒度向量数: {len(large_embeddings)}, 维度: {large_embeddings.shape[1]}")

        temp_store_large = FAISSVectorStore(self.embedding_model.dim)
        large_embeddings, dense_large = temp_store_large.clean_embeddings(large_embeddings, dense_large)

        print(f"\n构建FAISS索引...")
        self.small_vector_store = FAISSVectorStore(self.embedding_model.dim)
        self.small_vector_store.add_embeddings(small_embeddings, dense_small)
        print(f"小粒度FAISS索引: {self.small_vector_store.get_total_count()} 条向量")

        self.large_vector_store = FAISSVectorStore(self.embedding_model.dim)
        self.large_vector_store.add_embeddings(large_embeddings, dense_large)
        print(f"大粒度FAISS索引: {self.large_vector_store.get_total_count()} 条向量")

        self.small_vector_store.save("small")
        self.small_vector_store.save_embeddings(small_embeddings, "small")
        self.large_vector_store.save("large")
        self.large_vector_store.save_embeddings(large_embeddings, "large")
        print("向量存储已保存到缓存")

        self.small_dense_embeddings = small_embeddings
        self.large_dense_embeddings = large_embeddings

        print(f"\n--- Stage 3b: Q&A知识点加载 ---")
        self.qa_pairs = self.qa_extractor.extract_from_chunks(filtered_small)
        qa_path = self.qa_extractor.save_qa_pairs(self.qa_pairs)
        print(f"Q&A知识点: {len(self.qa_pairs)} 对（AI生成 + 多轮核查筛选）")
        print(f"Q&A存储路径: {qa_path}")

        type_counts = {}
        doc_counts = {}
        for qa in self.qa_pairs:
            t = qa.get("qa_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            d = qa.get("source_doc", "unknown")
            doc_counts[d] = doc_counts.get(d, 0) + 1
        print(f"  按类型:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {t}: {c} 对")
        print(f"  按文档: {len(doc_counts)} 个文档，每文档 {min(doc_counts.values())}-{max(doc_counts.values())} 对")

        self.qa_retriever = QARetriever(self.qa_pairs, embedding_model=self.embedding_model)

        self.retriever = HybridRetriever(
            embedding_model=self.embedding_model,
            small_vector_store=self.small_vector_store,
            large_vector_store=self.large_vector_store,
            all_chunks=all_chunks,
            small_dense_chunks=dense_small,
            large_dense_chunks=dense_large,
            small_dense_embeddings=small_embeddings,
            large_dense_embeddings=large_embeddings,
            qa_retriever=self.qa_retriever,
        )
        print(f"\n检索器初始化完成（Dense小+Dense大+BM25+QA四路检索 + Cross-Encoder精排）")

        print(f"\n{'='*60}")
        print(f"  构建完成汇总")
        print(f"{'='*60}")
        print(f"  文档数: {len(documents)}")
        print(f"  精炼文本量: {total_refined // 1000}K")
        print(f"  小粒度chunk: {len(filtered_small)}")
        print(f"  大粒度chunk: {len(filtered_large)}")
        print(f"  小粒度Dense向量: {len(dense_small)}")
        print(f"  大粒度Dense向量: {len(dense_large)}")
        print(f"  Q&A知识点: {len(self.qa_pairs)} 对")
        print(f"{'='*60}\n")

    def _check_cache(self) -> bool:
        small_index = os.path.join(VECTOR_STORE_DIR, "small", "faiss.index")
        small_meta = os.path.join(VECTOR_STORE_DIR, "small", "metadata.json")
        large_index = os.path.join(VECTOR_STORE_DIR, "large", "faiss.index")
        large_meta = os.path.join(VECTOR_STORE_DIR, "large", "metadata.json")

        small_ok = os.path.exists(small_index) and os.path.exists(small_meta)
        large_ok = os.path.exists(large_index) and os.path.exists(large_meta)
        return small_ok and large_ok

    def _load_from_cache(self):
        self.embedding_model = EmbeddingModel()

        self.small_vector_store = FAISSVectorStore(self.embedding_model.dim)
        self.small_vector_store.load("small")
        dense_small = [self.small_vector_store.id_to_metadata[i]
                       for i in sorted(self.small_vector_store.id_to_metadata.keys())]
        try:
            small_embeddings = self.small_vector_store.load_embeddings("small")
        except FileNotFoundError:
            small_embeddings = None

        self.large_vector_store = FAISSVectorStore(self.embedding_model.dim)
        self.large_vector_store.load("large")
        dense_large = [self.large_vector_store.id_to_metadata[i]
                       for i in sorted(self.large_vector_store.id_to_metadata.keys())]
        try:
            large_embeddings = self.large_vector_store.load_embeddings("large")
        except FileNotFoundError:
            large_embeddings = None

        all_chunks = dense_small + dense_large
        self.all_chunks = all_chunks
        self.small_dense_embeddings = small_embeddings
        self.large_dense_embeddings = large_embeddings

        self.qa_pairs = self.qa_extractor.load_qa_pairs()
        self.qa_retriever = QARetriever(self.qa_pairs, embedding_model=self.embedding_model)

        self.retriever = HybridRetriever(
            embedding_model=self.embedding_model,
            small_vector_store=self.small_vector_store,
            large_vector_store=self.large_vector_store,
            all_chunks=all_chunks,
            small_dense_chunks=dense_small,
            large_dense_chunks=dense_large,
            small_dense_embeddings=small_embeddings,
            large_dense_embeddings=large_embeddings,
            qa_retriever=self.qa_retriever,
        )
        print(f"缓存加载完成，小粒度Dense {len(dense_small)} 条，大粒度Dense {len(dense_large)} 条，BM25全量 {len(all_chunks)} 条，Q&A {len(self.qa_pairs)} 对（AI生成+核查筛选）\n")

    def search(self, query: str, top_k=FINAL_TOP_K) -> Dict:
        if self.retriever is None:
            raise RuntimeError("请先调用 build_knowledge_base() 构建知识库")

        start_time = time.time()
        result = self.retriever.search(
            query,
            dense_top_k=DENSE_TOP_K,
            bm25_top_k=BM25_TOP_K,
            qa_top_k=QA_TOP_K,
            final_top_k=top_k,
            use_query_expansion=True,
        )
        elapsed = (time.time() - start_time) * 1000
        result["retrieval_time_ms"] = round(elapsed, 2)

        if self.retriever and self.retriever._cross_encoder:
            result["cross_encoder_stats"] = self.retriever._cross_encoder.get_stats()

        merged_context = self.formatter.format_context(result)
        result["merged_context"] = merged_context

        return result

    def query(self, question: str):
        print("\n" + "=" * 60)
        print(f"查询问题：{question}")
        print("=" * 60)

        result = self.search(question)
        formatted = self.formatter.format_full_result(result)
        print(formatted)

        json_path = self.formatter.save_result(result)
        print(f"检索结果已保存至: {json_path}")

        return result


def main():
    pipeline = RAGPipeline()
    pipeline.build_knowledge_base()

    for q in TEST_QUESTIONS:
        pipeline.query(q)


if __name__ == "__main__":
    main()
