# 管理类知识库 RAG 系统

基于七阶段精炼管道的管理学知识库检索增强生成系统，纯 Python 代码实现。

## 核心流程

```
原始知识库
  │
  ▼ Stage 1: 数据清洗
  │  · 文档名规范化（清除 z-library 水印、ISBN 等噪声）
  │  · 繁简统一（法則→法则 等）
  │  · jieba 分词注入 100+ 管理学领域词
  │  · 内容精炼：长文档压缩至 30K 字符（保留定义/框架/结论段落）
  │
  ▼ Stage 2: 文本切分 + chunk 级质量过滤
  │  · 细粒度：600 字/块，重叠 100 字
  │  · 粗粒度：1200 字/块，重叠 200 字
  │  · 质量过滤：移除纯案例、低术语密度、过短 chunk
  │  · Dense 采样：按章节均匀采样，保证每文档覆盖
  │
  ▼ Stage 3a: 向量化 + 再次清洗筛选向量
  │  · BGE-large-zh-v1.5 编码（1024 维，C-MTEB 检索 70.23）
  │  · 向量去重：余弦相似度 > 0.90 的冗余向量移除
  │  · 质量筛选：综合唯一性(40%) + 邻居密度(30%) + 文本质量(30%)，
  │    移除 bottom 8%
  │  · 存入 FAISS 向量数据库
  │
  ├─ Stage 3b: Q&A 知识点构建
  │  · 人工构建六类问答对：定义/框架/原则/对比/事实/枚举
  │  · 多轮质量核查：问题-答案匹配度、准确性、完整性、冗余检测
  │  · 覆盖15份文档，390对高质量问答对
  │
  ▼ Stage 4: 四路混合检索 + 精排
     · 查询扩展：问题类型感知扩展（定义/对比/方法/原因/案例）
     · Dense-细(BGE 600字) + Dense-粗(BGE 1200字) + BM25(章节boost) + Q&A
     · RRF 融合 → MMR 动态参数重排 → Cross-Encoder精排 → 来源去重 → 父文档还原
     · 输出：Top-K 相关片段 + 来源文档标注 + Cross-Encoder分数
```

## 环境要求

- **Python 3.12**（faiss-cpu 不支持 3.13+）
- **向量数据库**：FAISS（faiss-cpu）

## 仓库清理记录

- **已删除**: 顶层运行时缓存 `__pycache__` 中的编译文件和其他临时/编辑器产生的文件（如存在）。
- **保留**: `python312/` 内置解释器与 `models/BAAI/` 下的模型缓存按要求保留，不做删除。

如需恢复已删除项，请参考项目说明中的环境与模型获取部分或联系仓库管理员以重新下载模型文件。
> 项目内置 Python 3.12：`python312/python.exe`，已预装全部依赖，推荐直接使用。

### 依赖安装

若不用内置解释器，请自行安装 Python 3.12：

```bash
# 推荐：使用项目内置 Python 3.12
python312/python.exe -m pip install -r requirements.txt

# 或使用系统 Python 3.12
python -m pip install -r requirements.txt

# Windows 上 torch 必须用 CPU 版：
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

或使用虚拟环境：

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 缓存机制

> 向量化阶段（BGE-large-zh-v1.5 编码 14 份文档的所有 chunk）在 CPU 上耗时约 40-50 分钟。
> 系统默认行为是：**检测到缓存后直接加载，跳过向量化阶段**，启动时间从 50 分钟缩短到约 30 秒。
> 只有显式传入 `--force-rebuild` 参数时才会重新执行全量向量化。

### 缓存存储位置

```
项目根目录/
├── data/
│   ├── vector_store/          ← FAISS 向量缓存
│   │   ├── small/             ← 细粒度索引
│   │   └── large/             ← 粗粒度索引
│   ├── cross_encoder_cache/   ← Cross-Encoder 精排结果缓存
│   └── qa_store/
│       └── qa_pairs_final.json  ← 390 对 Q&A 知识点
└── models/
    ├── BAAI/bge-large-zh-v1___5/  ← BGE 向量模型本地缓存
    └── BAAI/bge-reranker-large/   ← Cross-Encoder 精排模型本地缓存
```

## 快速启动

```bash
# 首次运行：构建知识库（CPU 约 40-50 分钟，完成后自动保存缓存）
python -m src.core.pipeline

# 后续运行：自动加载缓存（约 30 秒）
python -m src.core.pipeline

# 评估：Step 1 收集检索结果
python scripts/collect_retrieval_results.py

# 评估：Step 3 计算指标（Step 2 由AI助手完成）
python -m src.evaluation.eval_quick   # Quick模式
python -m src.evaluation.eval_full    # Full模式

# 强制重建缓存
python -m src.core.pipeline --force-rebuild
```

## 项目结构

```
├── data/
│   ├── knowledge_base/       # 原始知识库文档（15 份管理学著作）
│   ├── vector_store/         # FAISS 向量缓存（自动生成）
│   │   ├── small/            # 细粒度向量索引
│   │   └── large/            # 粗粒度向量索引
│   ├── qa_store/             # Q&A 知识点存储
│   │   └── qa_pairs_final.json  # 390 对高质量问答对
│   ├── eval/                 # 评估数据
│   │   ├── retrieval_results.json    # 检索结果
│   │   └── ragas_evaluation.json     # AI评估结果
│   └── output/               # 评估报告输出
├── models/                   # 模型本地缓存
│   ├── BAAI/bge-large-zh-v1___5/    # BGE 向量模型
│   └── BAAI/bge-reranker-large/     # Cross-Encoder 精排模型
├── src/
│   ├── core/
│   │   ├── config.py         # 全局参数配置
│   │   └── pipeline.py       # 七阶段管道主入口
│   ├── indexing/
│   │   ├── text_splitter.py  # 数据清洗 + 文本切分 + chunk 质量过滤
│   │   ├── embedding.py      # BGE-large-zh-v1.5 向量化
│   │   ├── vector_store.py   # FAISS 向量存储 + 向量去重 + 质量筛选
│   │   └── qa_extractor.py   # Q&A 知识点管理
│   ├── retrieval/
│   │   ├── retriever.py      # 四路混合检索 + RRF + MMR + Cross-Encoder精排
│   │   └── formatter.py      # 结果格式化 + 来源标注
│   ├── evaluation/
│   │   ├── eval_quick.py     # Quick 模式评估入口
│   │   ├── eval_full.py      # Full 模式评估入口
│   │   ├── dataset.py        # 检索结果和AI评估结果加载
│   │   ├── metrics.py        # RAGAS风格指标计算
│   │   └── reporter.py       # 报告生成
│   └── utils/
│       ├── constants.py      # 统一常量（STOP_WORDS, QUESTION_TYPE_PATTERNS）
│       ├── logger.py         # 统一日志系统
│       ├── path_utils.py     # 路径校验与工具
│       └── text_processing.py # 文本处理工具
├── scripts/
│   └── collect_retrieval_results.py  # 评估Step1: 收集检索结果
├── docs/
│   ├── 评估方案.md                    # RAGAS风格评估方案
│   └── 评估系统历史变更与错误记录.md    # 历史变更记录
├── requirements.txt
└── README.md
```

## 评估方法

采用 RAGAS 风格的 reference-free 评估，无需金标数据集：

1. **收集检索结果**：`python scripts/collect_retrieval_results.py`
2. **AI语义评估**：由AI助手对每个query-chunk对判断相关性
3. **计算指标**：Context Precision、Context Relevance、Context Sufficiency、Ranking Quality、NDCG

详见 [docs/评估方案.md](docs/评估方案.md)

## 关键参数（src/core/config.py）

| 参数 | 值 | 说明 |
|------|------|------|
| CHUNK_SIZE_SMALL | 600 | 细粒度 chunk 大小 |
| CHUNK_SIZE_LARGE | 1200 | 粗粒度 chunk 大小 |
| CHUNK_OVERLAP_SMALL | 100 | 细粒度重叠字符数 |
| CHUNK_OVERLAP_LARGE | 200 | 粗粒度重叠字符数 |
| PARENT_CHUNK_SIZE | 2000 | 父文档窗口大小 |
| DENSE_TOP_K | 50 | Dense 检索候选数 |
| BM25_TOP_K | 50 | BM25 检索候选数 |
| QA_TOP_K | 15 | Q&A 检索候选数 |
| MMR_LAMBDA | 0.45 | MMR 相关性权重（动态调整） |
| FINAL_TOP_K | 10 | 最终返回结果数 |
| MAX_CHUNKS_PER_DOC | 5 | 每文档最大返回 chunk 数 |
| USE_CROSS_ENCODER | True | Cross-Encoder 精排开关 |
| BM25_CHAPTER_BOOST | 0.15 | BM25 章节标题匹配加成系数 |

## 检索优化特性

### Cross-Encoder 精排（默认启用，带查询级缓存）

- 默认已启用 `BAAI/bge-reranker-large` 精排
- 配有查询级缓存，首次推理后结果自动落盘，重复查询秒级返回
- 如需关闭，在 `src/core/config.py` 中设置 `USE_CROSS_ENCODER = False`

```
检索管道：查询 → 四路检索 → RRF融合 → MMR重排 → Cross-Encoder精排 → 最终结果
```

### 查询扩展增强

根据问题类型自动扩展查询词：
- **定义类**：追加"概念 定义 含义"
- **对比类**：追加"异同 对比分析 区别"
- **方法类**：追加"步骤 流程 方法"
- **原因类**：追加"原因 因素 影响"

### BM25 章节标题 Boost

对章节标题与查询词重叠的 chunk 给予分数加成（1 + overlap_count × 0.15）。

### MMR 动态参数

根据问题类型动态调整 MMR 的 λ 参数：
- 对比类问题：λ=0.30（更高多样性）
- 定义类问题：λ=0.60（更高相关性）
- 默认：λ=0.45（平衡模式）

## 作业要求覆盖

### 基础得分（75分）

| 检查项 | 要求 | 覆盖 | 说明 |
|--------|------|------|------|
| 实现形式 | 低代码平台 | ✅ 加分 | 纯 Python 代码实现，无低代码平台依赖 |
| 知识库≥10份 | 至少10份文档 | ✅ | 15 份管理学文档全部成功加载 |
| 数据清洗与切分 | 文本提取+清洗+切分 | ✅ | Stage 1+2：噪声清除+精炼+双粒度切分+质量过滤 |
| 检索链路 | 输入问题→检索Top-K片段 | ✅ | Stage 4：四路混合检索 + RRF + MMR + Cross-Encoder精排 |
| 生成链路 | 接入大模型+Prompt模板 | ⚠️ 下游 | 本系统负责检索输出，生成由下游 LLM 完成 |
| 系统产出 | 无报错完成全过程 | ✅ | 缓存加载约 30 秒即可运行 |
| 输出含来源依据 | 回答+参考文档/片段 | ✅ | 每条结果标注来源文档+章节+chunk 位置 |

### 加分项（25分）

| 检查项 | 要求 | 覆盖 | 说明 |
|--------|------|------|------|
| 代码实现 | 完全代码实现 | ✅ | 纯 Python |
| Q&A 对构建 | 提取高质量问答对 | ✅ | 390 对高质量 Q&A（15文档覆盖，6种qa_type） |
| 多文档横向对比 | 跨文档问题检索 | ✅ | 四路检索天然跨文档，7 道 comparison 题评估 |
| 控制变量分析 | 对比不同参数影响 | ✅ | 评估框架支持，具体实验由下游完成 |
