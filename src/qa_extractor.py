import re
import json
import os
import numpy as np
from typing import List, Dict, Optional, Set
from src.config import QA_STORE_DIR, QA_SEMANTIC_WEIGHT, QA_BM25_WEIGHT


class QAExtractor:
    """QA对提取器：正则初步提取 + 多层级质量过滤"""

    # 定义类正则
    DEFINITION_PATTERNS = [
        (re.compile(r'(?:^|[，。；！？\n])([^，。；！？\n\s]{2,20}?)是指([^。；！？\n]{10,200})[。；]'), "definition"),
        (re.compile(r'(?:^|[，。；！？\n])([^，。；！？\n\s]{2,20}?)定义为([^。；！？\n]{10,200})[。；]'), "definition"),
        (re.compile(r'(?:^|[，。；！？\n])所谓([^，。；！？\n\s]{2,20}?)[，,]就是([^。；！？\n]{10,200})[。；]'), "definition"),
        (re.compile(r'(?:^|[，。；！？\n])([^，。；！？\n\s]{2,20}?)的核心是([^。；！？\n]{10,200})[。；]'), "definition"),
        (re.compile(r'(?:^|[，。；！？\n])([^，。；！？\n\s]{2,20}?)的本质是([^。；！？\n]{10,200})[。；]'), "definition"),
        (re.compile(r'(?:^|[，。；！？\n])([^，。；！？\n\s]{2,20}?)指的是([^。；！？\n]{10,200})[。；]'), "definition"),
    ]

    FRAMEWORK_PATTERNS = [
        (re.compile(r'(?:^|[，。；！？\n])([^，。；！？\n\s]{2,20}?)包括以下(?:几个)?(?:方面|部分|要素|维度|类型|阶段)[：:]([^。；！？\n]{15,300})[。；]'), "framework"),
        (re.compile(r'(?:^|[，。；！？\n])([^，。；！？\n\s]{2,20}?)主要(?:有|包括|分为|包含)[：:]?([^。；！？\n]{15,300})[。；]'), "framework"),
        (re.compile(r'(?:^|[，。；！？\n])([^，。；！？\n\s]{2,20}?)分为([^。；！？\n]{15,300})[。；]'), "framework"),
        (re.compile(r'(?:^|[，。；！？\n])([^，。；！？\n\s]{2,20}?)由([^，。；！？\n]{5,30})组成'), "framework"),
        (re.compile(r'(?:^|[，。；！？\n])([^，。；！？\n\s]{2,20}?)由([^，。；！？\n]{5,30})构成'), "framework"),
    ]

    PRINCIPLE_PATTERNS = [
        (re.compile(r'(?:^|[，。；！？\n])([^，。；！？\n\s]{2,20}?)的原则是([^。；！？\n]{10,200})[。；]'), "principle"),
        (re.compile(r'(?:^|[，。；！？\n])([^，。；！？\n\s]{2,20}?)应遵循([^。；！？\n]{10,200})[。；]'), "principle"),
        (re.compile(r'(?:^|[，。；！？\n])([^，。；！？\n\s]{2,20}?)的基本原则(?:包括|有)[：:]?([^。；！？\n]{10,200})[。；]'), "principle"),
    ]

    COMPARISON_PATTERNS = [
        (re.compile(r'(?:^|[，。；！？\n])([^，。；！？\n\s]{2,20}?)与([^，。；！？\n\s]{2,20}?)的区别(?:在于|是)[：:]?([^。；！？\n]{10,200})[。；]'), "comparison"),
        (re.compile(r'(?:^|[，。；！？\n])([^，。；！？\n\s]{2,20}?)不同于([^，。；！？\n\s]{2,20}?)[，,]([^。；！？\n]{10,200})[。；]'), "comparison"),
    ]

    QUESTION_TEMPLATES = {
        "definition": "什么是{subject}？",
        "framework": "{subject}包含哪些内容？",
        "enumeration": "{subject}包含哪些要点/部分？",
        "principle": "{subject}的原则是什么？",
        "comparison": "{subject_a}和{subject_b}有什么区别？",
    }

    # 低质量subject前缀（句子碎片）
    BAD_PREFIXES = {
        '将', '把', '以', '并', '而', '且', '或', '但', '却', '还', '又', '也', '就', '都',
        '要', '会', '能', '可', '应', '该', '在', '对', '从', '到', '向', '为', '被', '让',
        '给', '比', '与', '和', '同', '如', '若', '即', '则', '故', '因', '所', '这', '那',
        '此', '其', '之', '者', '个', '种', '类', '项', '第', '本', '该', '某', '各', '每',
        '凡', '我', '你', '他', '她', '它', '们', '的', '了', '着', '过', '是', '有', '不',
        '没', '无', '非', '未', '否', '别', '很', '太', '极', '最', '更', '较', '首先', '然后',
        '接着', '最后', '总之', '综上所述', '例如', '比如', '像', '正如', '犹如', '仿佛',
        '一方面', '另一方面', '其次', '再次', '一是', '二是', '三是', '第一', '第二', '第三',
        '①', '②', '③', '（', '(', '“', '"', '《', '这里', '那里', '哪里', '这样', '那样',
        '因此', '所以', '于是', '从而', '因而', '可见', '显然', '不仅', '不但', '虽然', '尽管',
        '即使', '即便', '除非', '除了', '除去', '无论', '不管', '不论', '由于', '因为', '鉴于',
        '基于', '根据', '按照', '随着', '通过', '对于', '关于', '至于', '以及', '还有',
        '也许', '或许', '可能', '大概', '大约', '似乎', '好像', '几乎', '简直', '根本',
        '完全', '绝对', '已经', '曾经', '刚刚', '正在', '将要', '就要', '马上', '立刻',
        '专门', '特意', '特别', '尤其', '格外', '分外', '非常', '十分', '越发', '更加',
        '越来越', '逐渐', '渐渐', '慢慢', '忽然', '突然', '猛然', '居然', '竟然',
        '果然', '果真', '自然', '当然', '固然', '然而', '但是', '可是', '不过', '只是',
        '何况', '况且', '再说', '或者', '或是', '还是', '要么', '既', '既然', '与其',
        '宁可', '宁愿', '宁肯', '一面', '一边', '一来', '二来', '一则', '一则',
        '从而构建', '在这', '在这本书', '在本书', '本书', '本章', '本节', '本章中',
        '本节中', '文章', '文中', '作者', '笔者', '我们', '大家', '读者', '人们',
        '有人', '有人认为', '有人认为', '有人认为', '有人认为',
    }

    # 低质量subject后缀
    BAD_SUFFIXES = {
        '的', '了', '着', '过', '是', '有', '在', '将', '把', '以', '并', '而', '且',
        '或', '但', '却', '还', '又', '也', '就', '都', '为', '被', '让', '给', '比',
        '与', '和', '同', '如', '若', '即', '则', '故', '因', '所', '之', '者', '个',
        '种', '类', '项', '等', '等等', '之一', '之一', '之一', '之一',
        '中', '上', '下', '里', '内', '外', '间', '边', '面', '头', '部', '方',
        '时', '时候', '期间', '过程', '阶段', '步骤', '方面', '部分', '领域',
        '角度', '层次', '水平', '程度', '范围', '规模', '标准', '原则', '基础',
        '条件', '情况', '状态', '形式', '方式', '方法', '手段', '途径', '渠道',
        '模式', '类型', '种类', '类别', '性质', '特点', '特征', '特色', '优势',
        '劣势', '作用', '功能', '意义', '价值', '目的', '目标', '任务', '职责',
        '责任', '权利', '权力', '利益', '效果', '结果', '后果', '影响', '关系',
        '联系', '区别', '差异', '变化', '发展', '进步', '提高', '降低', '增加',
        '减少', '扩大', '缩小', '加强', '减弱', '改善', '改进', '改革', '创新',
        '突破', '进展', '成就', '成功', '失败', '问题', '困难', '挑战', '机遇',
        '风险', '危机', '矛盾', '冲突', '斗争', '竞争', '合作', '协调', '配合',
        '支持', '帮助', '促进', '推动', '引导', '带领', '组织', '管理', '领导',
        '控制', '监督', '检查', '评估', '评价', '考核', '激励', '约束', '规范',
        '制度', '机制', '体制', '体系', '系统', '结构', '框架', '模型', '理论',
        '概念', '定义', '观点', '看法', '意见', '建议', '主张', '态度', '立场',
        '倾向', '偏好', '选择', '决定', '决策', '计划', '方案', '策略', '战略',
        '战术', '措施', '行动', '活动', '工作', '事业', '业务', '事务', '事情',
        '事件', '现象', '事实', '实例', '案例', '经验', '教训', '启示', '借鉴',
        '参考', '依据', '根据', '按照', '依照', '遵循', '遵守', '符合', '适应',
        '满足', '达到', '实现', '完成', '执行', '实施', '落实', '贯彻', '执行',
        '开展', '进行', '推进', '推动', '促进', '加强', '深化', '拓展', '延伸',
        '扩大', '提升', '提高', '增强', '增进', '增加', '增长', '发展', '进步',
        '改善', '改进', '完善', '优化', '调整', '转变', '改变', '改革', '创新',
        '突破', '超越', '领先', '优先', '重点', '核心', '关键', '根本', '基础',
        '前提', '保障', '支撑', '依托', '凭借', '利用', '运用', '使用', '采用',
        '采取', '实行', '实施', '执行', '履行', '承担', '负责', '主持', '主导',
        '主要', '重要', '重大', '关键', '核心', '根本', '基本', '主要', '首要',
        '第一', '主要', '重要', '重大', '关键', '核心', '根本', '基本',
    }

    # 问题质量检测：这些模式表示问题本身不完整/有碎片
    BAD_QUESTION_PATTERNS = [
        re.compile(r'^.+?的[是|原则|步骤|做法|方法].*?[是|为|包括].*?$'),
    ]

    def __init__(self):
        self.bad_prefixes = self.BAD_PREFIXES
        self.bad_suffixes = self.BAD_SUFFIXES

    def _is_good_subject(self, subject: str) -> bool:
        """判断subject是否为高质量名词短语"""
        s = subject.strip()
        if len(s) < 2 or len(s) > 20:
            return False
        # 检查前缀
        for prefix in self.bad_prefixes:
            if s.startswith(prefix):
                # 例外：某些前缀+名词是合理的
                if prefix in {'新', '大', '小', '高', '低', '老', '初', '副'}:
                    continue
                return False
        # 检查后缀
        for suffix in self.bad_suffixes:
            if s.endswith(suffix) and len(s) <= len(suffix) + 2:
                return False
        # 不能全是标点或数字
        if re.match(r'^[\d\s\W]+$', s):
            return False
        # 不能包含过多英文（排除混合碎片）
        if len(re.findall(r'[a-zA-Z]', s)) > len(s) * 0.5:
            return False
        # 不能是单个重复字
        if len(set(s)) == 1:
            return False
        return True

    def _is_good_answer(self, answer: str) -> bool:
        """判断answer是否高质量"""
        a = answer.strip()
        # Accept shorter answers when they are enumerations/lists
        if len(a) < 10:
            # but allow if it looks like a list (contains 、 or (1) or 1.)
            if re.search(r'[、；;\n]|\(\d+\)|\d+\.', a):
                # ensure at least two list items
                items = re.split(r'\s*(?:\(|（)?\d+[\)）]|[、;；\n]|\d+\.)\s*', a)
                items = [it for it in items if it.strip()]
                if len(items) >= 2:
                    return True
            return False
        # 不能以标点开头（说明是句子碎片）
        if a[0] in '，。；！？、：:':
            return False
        # 不能全是标点
        if re.match(r'^[\s\W]+$', a):
            return False
        return True

    def _clean_subject(self, subject: str) -> str:
        """清理subject中的碎片"""
        s = subject.strip()
        # 去除前导碎片词
        for prefix in sorted(self.bad_prefixes, key=len, reverse=True):
            if s.startswith(prefix):
                remainder = s[len(prefix):].strip()
                if len(remainder) >= 2:
                    s = remainder
                    break
        # 去除尾部不完整词
        for suffix in sorted(self.bad_suffixes, key=len, reverse=True):
            if s.endswith(suffix) and len(s) > len(suffix) + 2:
                s = s[:-len(suffix)].strip()
                break
        return s

    def _split_enumeration_items(self, text: str) -> List[str]:
        """将可能的枚举文本拆分为若干条目并清洗"""
        # 首先按常见分隔符拆分
        parts = re.split(r'\s*(?:；|;|\n)\s*', text)
        items = []
        for p in parts:
            # 按数字或序号分割，例如 (1)、1.、一、二、三、①
            sub = re.split(r'(?:(?:\(|（)?\d+[\)）]|[①-⑩]|[一二三四五六七八九十]+[、.])', p)
            for s in sub:
                s = s.strip(' .、;；:\t\r\n')
                if s:
                    items.append(s.strip())
        # 去重且保持顺序
        seen = set(); uniq = []
        for it in items:
            if it not in seen:
                seen.add(it); uniq.append(it)
        return uniq

    def _extract_enumerations(self, text: str) -> List[Dict]:
        """尝试从文本中提取带序号/枚举的段落并生成 QA"""
        results = []
        # 匹配形如“X分为...：...”“X包括以下...：...”等，捕获后面的枚举块
        enum_patterns = [
            re.compile(r'(?:^|[。；\n])([^，。；！？\n\s]{2,30}?(?:分为|分成|包括|包含|可分为)[：:]?)([\s\S]{6,400}?)(?:[。；\n]|$)') ,
            re.compile(r'(?:^|[。；\n])([^，。；！？\n\s]{2,30}?的?部分[：:]?)([\s\S]{6,400}?)(?:[。；\n]|$)') ,
        ]
        for pat in enum_patterns:
            for m in pat.finditer(text):
                subj = m.group(1).strip()
                body = m.group(2).strip()
                # try to extract enumerated items
                items = self._split_enumeration_items(body)
                if len(items) >= 2 and self._is_good_subject(subj):
                    answer_text = '；'.join([f'{i+1}. {it}' for i, it in enumerate(items)])
                    question = self.QUESTION_TEMPLATES.get('enumeration', '{subject}包含哪些内容？').format(subject=self._clean_subject(subj))
                    results.append({
                        'question': question,
                        'answer': answer_text,
                        'qa_type': 'enumeration',
                    })
        return results

    def extract_from_chunks(self, chunks: List[Dict]) -> List[Dict]:
        qa_pairs = []
        seen_questions = set()

        for chunk in chunks:
            content = chunk.get("content", "")
            source = chunk.get("source_doc", "")
            chapter = chunk.get("chapter", "")
            chunk_id = chunk.get("chunk_id", "")

            chunk_qa = []
            # 先尝试从段落中提取枚举/序号型内容
            chunk_qa.extend(self._extract_enumerations(content))
            chunk_qa.extend(self._extract_by_patterns(content, self.DEFINITION_PATTERNS, "definition"))
            chunk_qa.extend(self._extract_by_patterns(content, self.FRAMEWORK_PATTERNS, "framework"))
            chunk_qa.extend(self._extract_by_patterns(content, self.PRINCIPLE_PATTERNS, "principle"))
            chunk_qa.extend(self._extract_comparison(content))

            for qa in chunk_qa:
                q_normalized = re.sub(r'\s+', '', qa["question"])
                if q_normalized in seen_questions:
                    continue
                seen_questions.add(q_normalized)
                qa_pairs.append({
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "qa_type": qa["qa_type"],
                    "source_doc": source,
                    "chapter": chapter,
                    "chunk_id": chunk_id,
                })

        return qa_pairs

    def _extract_by_patterns(self, text: str, patterns: List[tuple], qa_type: str) -> List[Dict]:
        results = []
        for pattern, _ in patterns:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) >= 2:
                    subject = groups[0].strip()
                    answer_text = groups[-1].strip()

                    # 质量过滤
                    if not self._is_good_subject(subject):
                        # 尝试清理
                        cleaned = self._clean_subject(subject)
                        if self._is_good_subject(cleaned):
                            subject = cleaned
                        else:
                            continue

                    if not self._is_good_answer(answer_text):
                        continue

                    question = self.QUESTION_TEMPLATES[qa_type].format(subject=subject)
                    results.append({
                        "question": question,
                        "answer": answer_text,
                        "qa_type": qa_type,
                    })
        return results

    def _extract_comparison(self, text: str) -> List[Dict]:
        results = []
        for pattern, _ in self.COMPARISON_PATTERNS:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) >= 3:
                    subject_a = groups[0].strip()
                    subject_b = groups[1].strip()
                    answer_text = groups[2].strip()

                    if not self._is_good_subject(subject_a) or not self._is_good_subject(subject_b):
                        continue
                    if not self._is_good_answer(answer_text):
                        continue

                    question = self.QUESTION_TEMPLATES["comparison"].format(
                        subject_a=subject_a, subject_b=subject_b
                    )
                    results.append({
                        "question": question,
                        "answer": answer_text,
                        "qa_type": "comparison",
                    })
        return results

    def save_qa_pairs(self, qa_pairs: List[Dict], filename: str = "qa_pairs.json") -> str:
        os.makedirs(QA_STORE_DIR, exist_ok=True)
        filepath = os.path.join(QA_STORE_DIR, filename)
        # 在保存前为每条 QA 增加生成元数据（不改变外部格式）
        import datetime
        enriched = []
        for qa in qa_pairs:
            item = qa.copy()
            if 'gold_source' not in item:
                item['gold_source'] = 'pipeline_generated'
            item['generated_by'] = 'qa_extractor_v2'
            item['generated_at'] = datetime.datetime.utcnow().isoformat() + 'Z'
            enriched.append(item)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)
        return filepath

    def load_qa_pairs(self, filename: str = "qa_pairs.json") -> List[Dict]:
        filepath = os.path.join(QA_STORE_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return []


class QARetriever:
    def __init__(self, qa_pairs: List[Dict], embedding_model=None):
        self.qa_pairs = qa_pairs
        self.embedding_model = embedding_model
        self._question_embeddings = None
        self._build_index()

    def _build_index(self):
        import jieba
        from src.retriever import STOP_WORDS
        self.question_tokens = []
        for qa in self.qa_pairs:
            tokens = [w for w in jieba.cut(qa["question"]) if len(w.strip()) >= 2 and w.strip() not in STOP_WORDS]
            self.question_tokens.append(tokens)

        if self.embedding_model is not None and len(self.qa_pairs) > 0:
            questions = [qa["question"] for qa in self.qa_pairs]
            print(f"[QA-RETRIEVER] 使用BGE语义向量化 {len(questions)} 个问题")
            self._question_embeddings = self.embedding_model.encode_documents(
                questions, batch_size=64, show_progress=False
            )
            norms = np.linalg.norm(self._question_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._question_embeddings = (self._question_embeddings / norms).astype(np.float32)
            print(f"[QA-RETRIEVER] 问题向量维度: {self._question_embeddings.shape}")
        else:
            print(f"[QA-RETRIEVER] 无Embedding模型，使用纯BM25检索")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if not self.qa_pairs:
            return []

        bm25_scores = self._bm25_search(query)
        semantic_scores = self._semantic_search(query)

        if semantic_scores is not None:
            combined = QA_SEMANTIC_WEIGHT * semantic_scores + QA_BM25_WEIGHT * bm25_scores
        else:
            combined = bm25_scores

        top_indices = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if combined[idx] > 0:
                qa = self.qa_pairs[idx].copy()
                qa["score"] = float(combined[idx])
                qa["bm25_score"] = float(bm25_scores[idx])
                if semantic_scores is not None:
                    qa["semantic_score"] = float(semantic_scores[idx])
                results.append(qa)
        return results

    def _bm25_search(self, query: str) -> np.ndarray:
        import jieba
        from src.retriever import STOP_WORDS
        from rank_bm25 import BM25Okapi

        query_tokens = [w for w in jieba.cut(query) if len(w.strip()) >= 2 and w.strip() not in STOP_WORDS]
        if not query_tokens:
            return np.zeros(len(self.qa_pairs), dtype=np.float32)

        bm25 = BM25Okapi(self.question_tokens)
        scores = bm25.get_scores(query_tokens)
        scores = np.array(scores, dtype=np.float32)
        max_s = scores.max()
        if max_s > 0:
            scores = scores / max_s
        return scores

    def _semantic_search(self, query: str) -> Optional[np.ndarray]:
        if self._question_embeddings is None or self.embedding_model is None:
            return None

        query_emb = self.embedding_model.encode_query(query)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        query_emb = query_emb.astype(np.float32)

        sims = self._question_embeddings @ query_emb
        max_s = sims.max()
        if max_s > 0:
            sims = sims / max_s
        return sims
