# Pooling 评估流程：操作说明

## 整体流程（3 步）

```
Step 1: 生成候选池 CSV    →  你运行脚本，自动完成
Step 2: 人工标注 CSV      →  你手动填写（这是唯一需要你做的）
Step 3: 导入标注 + 评估   →  你运行脚本，自动完成
```

---

## Step 1: 生成候选池 CSV

### 运行命令

```bash
python scripts/generate_pool_for_annotation.py
```

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `data/eval/eval_dataset.json` | 评估数据集路径 |
| `--output` | `data/eval/annotation_pool.csv` | 输出 CSV 路径 |
| `--top-k` | `20` | 每路检索返回 top-K 数量 |

### 它做了什么

对每道评估题，运行 **4 路独立检索**：

1. **BM25** — 纯词频匹配，不依赖语义
2. **Dense-Small** — BGE 向量检索（小粒度 chunk）
3. **Dense-Large** — BGE 向量检索（大粒度 chunk）
4. **QA** — QA 知识点检索

再加上原始金标中的 chunk（标记为 `Original-Gold`），合并去重后导出 CSV。

### 输出文件

- `eval/eval/annotation_pool.csv` — 候选池，**你需要标注这个文件**
- `eval/eval/annotation_pool_stats.json` — 统计信息

### 预计耗时

约 5-10 分钟（取决于机器性能，主要是向量检索）

---

## Step 2: 人工标注 CSV

### 打开文件

用 Excel 或 WPS 打开 `eval_evat/annotation_pool.csv`

### CSV 列说明

| 列名 | 含义 | 谁填写 |
|------|------|--------|
| `query_id` | 题目 ID | 自动 |
| `query` | 问题文本 | 自动 |
| `chunk_id` | 候选 chunk 的 ID | 自动 |
| `source_doc` | chunk 所属文档 | 自动 |
| `chapter` | chunk 所属章节 | 自动 |
| `content_preview` | chunk 内容预览（前300字） | 自动 |
| `retrieval_sources` | 该 chunk 被哪些检索方法找到 | 自动 |
| `num_sources` | 被几路检索同时找到 | 自动 |
| `max_score` | 最高检索得分 | 自动 |
| `in_original_gold` | 是否在原始金标中（1=是，0=否） | 自动 |
| **`is_relevant`** | **该 chunk 是否与 query 相关** | **你填写** |
| **`relevance_level`** | **相关程度：高/中/低** | **你填写** |
| **`annotator`** | **标注者姓名** | **你填写** |
| **`notes`** | **备注** | **你填写** |

### 标注规则

**`is_relevant` 列**（最重要，必填）：

- 填 `1`：该 chunk 内容**直接回答了** query 的问题，或包含了 query 所需的关键信息
- 填 `0`：该 chunk 内容与 query **无关**，或只有极其间接的关联

**判断标准**：

1. 先看 `query`（问题），理解问题在问什么
2. 再看 `content_preview`（chunk 内容预览）
3. 问自己：**如果用户问了这个问题，这个 chunk 的内容能帮助回答吗？**
4. 如果能 → 填 `1`；如果不能 → 填 `0`

**`relevance_level` 列**（选填）：

- `高`：chunk 直接包含问题的答案
- `中`：chunk 包含部分相关信息
- `低`：chunk 只有间接关联

### 标注技巧

1. **按 query_id 分组标注**：先筛选同一个 query_id 的所有行，一起看
2. **优先看 `num_sources` 高的**：被多路检索同时找到的 chunk，更可能相关
3. **注意 `in_original_gold`**：原始金标中的 chunk 不一定都对，请独立判断
4. **不确定时填 0**：宁可漏标也不要误标，严格标准下评估更可信
5. **如果 content_preview 不够**：可以对照 `source_doc` 和 `chapter` 去知识库原文中查看

### 工作量估算

- 63 道题，每题约 50 个候选
- 总计约 3000+ 行
- 每行判断约 10-30 秒
- 预计总工作量：8-15 小时

### 标注示例

```
query_id: Q001
query: 管理的核心思想是什么

chunk_id: 认识管理_small_chunk_16
content_preview: "管理的核心是决策。决策贯穿于管理活动的全过程..."
→ is_relevant: 1  （直接回答了"管理的核心思想"）
→ relevance_level: 高

chunk_id: 基业长青_large_chunk_55
content_preview: "高瞻远瞩公司能够持续成功的根本原因..."
→ is_relevant: 0  （讲的是基业长青，不是管理的核心思想）
→ relevance_level: 低
```

---

## Step 3: 导入标注 + 评估

### 3a. 导入人工标注

```bash
python scripts/import_annotations.py --csv data/eval/annotation_pool.csv
```

这会生成 `data/eval/eval_dataset_human.json`，其中：
- `is_relevant=1` 的 chunk 成为新的金标
- `gold_source` 被设为 `"human"`
- 原始数据集不变

### 3b. 运行评估

```bash
# 严格模式：只用人工金标的题评估
python scripts/run_eval_with_human_gold.py --dataset data/eval/eval_dataset_human.json --mode strict

# 半严格模式：全部题都评估，但标记非人工金标的可信度
python scripts/run_eval_with_human_gold.py --dataset data/eval/eval_dataset_human.json --mode semi
```

### 3c. 查看报告

评估报告保存在 `data/output/eval_human_gold_strict_report.json`

---

## 辅助工具

### 数据一致性校验

```bash
python scripts/check_dataset_consistency.py --dataset data/eval/eval_dataset.json
```

检查金标来源分布、chunk_id 是否存在、文档名是否匹配等。

### 旧版评估（带警告）

```bash
python src/eval_full.py
```

现在运行旧版评估会自动检测金标来源，如果发现 `pipeline_generated` 会打印警告。

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `scripts/generate_pool_for_annotation.py` | 生成候选池 CSV |
| `scripts/import_annotations.py` | 导入人工标注结果 |
| `scripts/run_eval_with_human_gold.py` | 基于人工金标的评估 |
| `scripts/check_dataset_consistency.py` | 数据一致性校验 |
| `data/eval/annotation_pool.csv` | 候选池（你标注这个） |
| `data/eval/eval_dataset_human.json` | 人工标注后的数据集 |
| `data/output/eval_human_gold_strict_report.json` | 评估报告 |

---

## 常见问题

**Q: 我能只标注部分题目吗？**
A: 可以。未标注的题在导入时会被跳过，保持原始金标不变。建议至少标注 20 道题后再运行评估。

**Q: 标注到一半可以中断吗？**
A: 可以。保存 CSV 即可，下次继续。导入脚本只会处理 `is_relevant` 列非空的行。

**Q: 多人可以同时标注吗？**
A: 可以。每人标注不同的 query_id 范围，最后合并 CSV。用 `annotator` 列区分标注者。

**Q: 标注完发现标错了怎么办？**
A: 直接修改 CSV 中的 `is_relevant` 值，重新运行 `import_annotations.py` 即可。
