import os

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# merged_context最大长度限制（下游LLM上下文窗口兼容）
MAX_MERGED_CONTEXT_LENGTH: int = 4000

KNOWLEDGE_BASE_DIR: str = os.path.join(BASE_DIR, "data", "knowledge_base")
OUTPUT_DIR: str = os.path.join(BASE_DIR, "data", "output")
VECTOR_STORE_DIR: str = os.path.join(BASE_DIR, "data", "vector_store")
QA_STORE_DIR: str = os.path.join(BASE_DIR, "data", "qa_store")

CHUNK_SIZE_SMALL: int = 600
CHUNK_SIZE_LARGE: int = 1200
CHUNK_OVERLAP_SMALL: int = 100
CHUNK_OVERLAP_LARGE: int = 200
PARENT_CHUNK_SIZE: int = 2000

DENSE_MAX_CHUNKS: int = 500

REFINER_MAX_CHARS_PER_DOC: int = 30000

EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-zh-v1.5"
EMBEDDING_LOCAL_PATH: str = os.path.join(BASE_DIR, "models", "BAAI", "bge-large-zh-v1___5")
EMBEDDING_QUERY_INSTRUCTION: str = "为这个句子生成表示以用于检索相关文章："
EMBEDDING_DIM: int = 1024

DENSE_TOP_K: int = 50
BM25_TOP_K: int = 50
QA_TOP_K: int = 15
QA_SEMANTIC_WEIGHT: float = 0.7
QA_BM25_WEIGHT: float = 0.3
RRF_K: int = 60
DENSE_SMALL_WEIGHT: float = 1.5
DENSE_LARGE_WEIGHT: float = 1.2
BM25_WEIGHT: float = 1.0
QA_WEIGHT: float = 1.3

MMR_LAMBDA: float = 0.45

FINAL_TOP_K: int = 15  # 最终返回chunk数，增大以提升覆盖率
MAX_CHUNKS_PER_DOC: int = 5

QUERY_EXPAND_COUNT: int = 4

VECTOR_DEDUP_THRESHOLD: float = 0.90
VECTOR_QUALITY_FILTER_RATIO: float = 0.08
VECTOR_MIN_PER_DOC: int = 5

MIN_CHUNK_LENGTH: int = 30

USE_CROSS_ENCODER: bool = True  # Cross-Encoder精排：默认启用，配有查询级缓存避免重复推理
CROSS_ENCODER_MODEL: str = "BAAI/bge-reranker-large"
CROSS_ENCODER_LOCAL_PATH: str = os.path.join(BASE_DIR, "models", "BAAI", "bge-reranker-large")
CROSS_ENCODER_TOP_K: int = 15
CROSS_ENCODER_BATCH_SIZE: int = 8
CROSS_ENCODER_CACHE_ENABLED: bool = True
CROSS_ENCODER_CACHE_DIR: str = os.path.join(BASE_DIR, "data", "cross_encoder_cache")

TEST_QUESTIONS: list = [
    "管理类核心思想是什么",
    "什么是战略管理中的SWOT分析",
    "人力资源管理包括哪些核心模块",
    "泰勒科学管理理论的核心观点是什么",
    "组织文化的层次是什么",
    "科学管理理论是谁提出的",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(QA_STORE_DIR, exist_ok=True)