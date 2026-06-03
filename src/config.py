import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# merged_context最大长度限制（下游LLM上下文窗口兼容）
MAX_MERGED_CONTEXT_LENGTH = 4000

KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
QA_STORE_DIR = os.path.join(BASE_DIR, "qa_store")

CHUNK_SIZE_SMALL = 600
CHUNK_SIZE_LARGE = 1200
CHUNK_OVERLAP_SMALL = 100
CHUNK_OVERLAP_LARGE = 200
PARENT_CHUNK_SIZE = 2000

DENSE_MAX_CHUNKS = 500

REFINER_MAX_CHARS_PER_DOC = 30000

EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
EMBEDDING_LOCAL_PATH = os.path.join(BASE_DIR, "models", "BAAI", "bge-large-zh-v1___5")
EMBEDDING_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
EMBEDDING_DIM = 1024

DENSE_TOP_K = 50
BM25_TOP_K = 50
QA_TOP_K = 15
QA_SEMANTIC_WEIGHT = 0.7
QA_BM25_WEIGHT = 0.3
RRF_K = 60
DENSE_SMALL_WEIGHT = 1.5
DENSE_LARGE_WEIGHT = 1.2
BM25_WEIGHT = 1.0
QA_WEIGHT = 1.3

MMR_LAMBDA = 0.45

FINAL_TOP_K = 15  # 最终返回chunk数，增大以提升覆盖率
MAX_CHUNKS_PER_DOC = 5

QUERY_EXPAND_COUNT = 4

VECTOR_DEDUP_THRESHOLD = 0.90
VECTOR_QUALITY_FILTER_RATIO = 0.08
VECTOR_MIN_PER_DOC = 5

MIN_CHUNK_LENGTH = 30

USE_CROSS_ENCODER = False  # Cross-Encoder精排：CPU上耗时极长（63题约1-2小时），默认关闭；GPU环境可开启
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-large"
CROSS_ENCODER_TOP_K = 15
CROSS_ENCODER_BATCH_SIZE = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(QA_STORE_DIR, exist_ok=True)
