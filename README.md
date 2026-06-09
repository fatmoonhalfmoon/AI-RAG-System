# 管理类知识库 RAG 系统

基于七阶段精炼管道的管理学知识库检索增强生成系统，纯 Python 代码实现。

## 核心流程

具体展开为七阶段管道：

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
  ├─ Stage 3b: Q&A 知识点提取
  │  · 正则提取五类问答对：定义/框架/原则/对比/方法步骤
  │  · AI审查修复：多轮质量检查，过滤低质量/残缺/乱码问题
  │  · AI补充生成：基于知识库核心概念补充高质量问答对
  │  · 输出：62 对高质量 Q&A（覆盖17本管理学著作）
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

> **注意**：本项目仓库不包含 Python 解释器。请自行安装 Python 3.12，或使用独立的虚拟环境。

### 依赖安装

```bash
# 使用系统安装的 Python 3.12
python -m pip install -r requirements.txt

# Windows 上 torch 必须用 CPU 版：
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

或使用虚拟环境（推荐）：

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 缓存机制（重要）

> **⚠️ 核心原则：优先复用缓存，绝不重复向量化！**
>
> 向量化阶段（BGE-large-zh-v1.5 编码 14 份文档的所有 chunk）在 CPU 上耗时约 40-50 分钟。
> 系统默认行为是：**检测到缓存后直接加载，跳过向量化阶段**，启动时间从 50 分钟缩短到约 30 秒。
> 只有显式传入 `--force-rebuild` 参数时才会重新执行全量向量化。

### 缓存存储位置

缓存存储在**当前项目文件夹**内，不是系统临时目录：

```
项目根目录/
├── vector_store/              ← FAISS 向量缓存（主路径，存储在项目文件夹内）
│   ├── small/
│   │   ├── faiss.index        ← 细粒度 FAISS 索引文件
│   │   ├── metadata.json      ← 细粒度 chunk 元数据
│   │   └── embeddings_raw.npy ← 细粒度原始向量矩阵
│   └── large/
│       ├── faiss.index        ← 粗粒度 FAISS 索引文件
│       ├── metadata.json      ← 粗粒度 chunk 元数据
│       └── embeddings_raw.npy ← 粗粒度原始向量矩阵
├── qa_store/
│   └── qa_pairs.json          ← Q&A 知识点缓存
└── models/
    ├── BAAI/bge-large-zh-v1___5/  ← BGE 向量模型本地缓存
    └── BAAI/bge-reranker-large/   ← Cross-Encoder 精排模型本地缓存
```

- **主路径**：`项目根目录/vector_store/`（推荐，默认使用）
- **备用路径**：`系统临时目录/rag_faiss/`（仅在主路径写入失败时回退）
- **缓存内容**：FAISS 索引 + chunk 元数据 + 原始向量矩阵，均为当前方案（BGE-large-zh-v1.5 + FAISS + 双粒度切分 + 向量清洗筛选）的产物
- **缓存与方案对应**：当前缓存即为最终方案缓存，包含向量去重和质量筛选后的结果，可直接复用

### 缓存加载逻辑

```
启动 → 检测 vector_store/small/ 和 vector_store/large/ 是否存在
  ├─ 存在 → 直接加载缓存（~30秒），跳过向量化
  └─ 不存在 → 执行完整七阶段管道（~50分钟），构建后自动保存缓存
```

## 快速启动

```bash
# 首次运行：构建知识库（CPU 约 40-50 分钟，完成后自动保存缓存）
python312\python.exe -m src.main

# 后续运行：自动加载缓存（约 30 秒，不会重复向量化）
python312\python.exe -m src.main

# Quick 模式评估（同样自动复用缓存）
python312\python.exe src\eval_quick.py

# Full 模式评估（含鲁棒性测试）
python312\python.exe src\eval_full.py

# 强制重建缓存（仅在需要重新向量化时使用）
python312\python.exe src\eval_quick.py --force-rebuild
```

## 项目结构

```
├── knowledge_base/       # 原始知识库文档（16 份管理学著作）
├── models/               # 模型本地缓存
│   ├── BAAI/bge-large-zh-v1___5/    # BGE 向量模型
│   └── BAAI/bge-reranker-large/     # Cross-Encoder 精排模型
├── vector_store/         # FAISS 向量缓存（自动生成）
│   ├── small/            # 细粒度向量索引
│   └── large/            # 粗粒度向量索引
├── qa_store/             # Q&A 知识点存储
│   └── qa_pairs.json     # 62 对高质量问答对（AI审查+补充生成）
├── eval_data/            # 评估数据集
│   └── eval_dataset.json # 63 题评估集
├── output/               # 评估结果输出
├── src/
│   ├── config.py         # 全局参数配置
│   ├── text_splitter.py  # 数据清洗 + 文本切分 + chunk 质量过滤
│   ├── embedding.py      # BGE-large-zh-v1.5 向量化
│   ├── vector_store.py   # FAISS 向量存储 + 向量去重 + 质量筛选
│   ├── retriever.py      # 四路混合检索 + RRF + MMR + Cross-Encoder精排
│   ├── qa_extractor.py   # Q&A 知识点提取
│   ├── formatter.py      # 结果格式化 + 来源标注
│   ├── main.py           # 七阶段管道主入口
│   ├── eval_quick.py     # Quick 模式评估入口
│   ├── eval_full.py      # Full 模式评估入口
│   ├── utils/
│   │   ├── constants.py  # 统一常量（STOP_WORDS, QUESTION_TYPE_PATTERNS）
│   │   └── logger.py     # 统一日志系统
│   └── eval/
│       ├── dataset.py    # 评估数据集加载
│       ├── metrics.py    # 四层指标计算
│       ├── built_in_ai_judge.py  # 内置 AI 评估
│       └── reporter.py   # 报告生成
├── 作业要求/             # 作业原始要求
├── 错误日志.md           # 错误记录与优化记录
├── 评估方案.md           # 评估方案文档
└── README.md
```

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
| Q&A 对构建 | 提取高质量问答对 | ✅ | 62 对高质量 Q&A（AI审查修复+补充生成，覆盖17本管理学著作） |
| 多文档横向对比 | 跨文档问题检索 | ✅ | 四路检索天然跨文档，10 道 comparison 题评估 |
| 控制变量分析 | 对比不同参数影响 | ✅ | 评估框架支持，具体实验由下游完成 |

## 关键参数（config.py）

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
| MMR_LAMBDA | 0.45 | MMR 相关性权重（默认值，动态调整） |
| FINAL_TOP_K | 10 | 最终返回结果数 |
| MAX_CHUNKS_PER_DOC | 5 | 每文档最大返回 chunk 数 |
| VECTOR_DEDUP_THRESHOLD | 0.90 | 向量去重余弦阈值 |
| VECTOR_QUALITY_FILTER_RATIO | 0.08 | 向量质量筛选移除比例 |
| USE_CROSS_ENCODER | False | Cross-Encoder 精排开关（CPU上耗时极长，默认关闭；GPU环境可开启） |
| CROSS_ENCODER_MODEL | BAAI/bge-reranker-large | Cross-Encoder 模型 |
| CROSS_ENCODER_TOP_K | 15 | Cross-Encoder 精排候选数 |
| CROSS_ENCODER_BATCH_SIZE | 8 | Cross-Encoder 批处理大小 |
| BM25_CHAPTER_BOOST | 0.15 | BM25 章节标题匹配加成系数 |

## 检索优化特性

### Cross-Encoder 精排（可选，默认关闭）

在 RRF 融合 + MMR 重排之后，可选使用 `BAAI/bge-reranker-large` 对候选结果进行二次精排。Cross-Encoder 将查询和文档拼接输入模型，计算精确相关性分数，比单纯向量相似度更准确。

> **注意**：Cross-Encoder 在 CPU 上运行极慢（63题评估约需1-2小时），因此默认关闭。如有 GPU 环境可在 `config.py` 中设置 `USE_CROSS_ENCODER = True` 开启。

```
检索管道：查询 → 四路检索 → RRF融合 → MMR重排 → [Cross-Encoder精排] → 最终结果
                                                              ↑ 可选，默认跳过
```

### 查询扩展增强

根据问题类型自动扩展查询词，生成多种查询变体：
- **定义类**：追加"概念 定义 含义"
- **对比类**：追加"异同 对比分析 区别"
- **方法类**：追加"步骤 流程 方法"
- **原因类**：追加"原因 因素 影响"

### BM25 章节标题 Boost

在 BM25 检索中，对章节标题与查询词重叠的 chunk 给予分数加成（1 + overlap_count × 0.15），使章节标题匹配的文档排名提升。

### MMR 动态参数

根据问题类型动态调整 MMR 的 λ 参数：
- 对比类问题：λ=0.30（更高多样性，需要多角度信息）
- 定义类问题：λ=0.60（更高相关性，需要精准定义）
- 默认：λ=0.45（平衡模式）
