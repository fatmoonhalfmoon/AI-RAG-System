"""
从知识库生成高质量 QA 对并写入 qa_store/qa_pairs.json

用法：
  python scripts/generate_qa_from_kb.py

说明：脚本不依赖Embedding模型，仅使用小粒度切分器和 `QAExtractor`。
"""
import os
import json
import shutil
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_splitter import ChineseTextSplitter, load_knowledge_base
from src.config import KNOWLEDGE_BASE_DIR
from src.qa_extractor import QAExtractor
from src.config import QA_STORE_DIR


def main():
    print("加载知识库并生成 QA 对...")
    documents = load_knowledge_base(KNOWLEDGE_BASE_DIR)
    if not documents:
        print("未找到知识库文档。请检查 knowledge_base/ 目录。")
        return

    splitter = ChineseTextSplitter.create_small_splitter()
    all_chunks = []
    for doc in documents:
        chunks = splitter.create_chunks_with_parents(doc['content'], doc['doc_name'])
        # 保留必要字段
        for c in chunks:
            all_chunks.append({
                'content': c.get('content', ''),
                'source_doc': c.get('source_doc', ''),
                'chapter': c.get('chapter', ''),
                'chunk_id': c.get('chunk_id', ''),
            })

    extractor = QAExtractor()
    qa_pairs = extractor.extract_from_chunks(all_chunks)

    # 备份旧文件
    out_path = os.path.join(QA_STORE_DIR, 'qa_pairs.json')
    if os.path.exists(out_path):
        bak = out_path + '.bak'
        shutil.copyfile(out_path, bak)
        print(f"已备份旧 QA 文件到: {bak}")

    saved = extractor.save_qa_pairs(qa_pairs)
    print(f"生成并保存 QA 对: {len(qa_pairs)} 对")
    print(f"保存路径: {saved}")


if __name__ == '__main__':
    main()
