import re
from typing import List, Set

import jieba

from src.utils.constants import STOP_WORDS


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    # 常见繁简或字符替换（可扩展）
    text = text.replace("法則", "法则")
    # 去除部分标点与特殊分隔符
    text = re.sub(r'[-_：:，,·\u3000]', '', text)
    # 规范空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str, min_len: int = 2, remove_stopwords: bool = True) -> List[str]:
    """统一中文分词接口，返回 token 列表。

    - 使用 jieba 切词
    - 过滤短词和停用词（可配置）
    """
    text = normalize_text(text)
    tokens = [w for w in jieba.cut(text) if w and w.strip()]
    if remove_stopwords:
        tokens = [w for w in tokens if len(w) >= min_len and w not in STOP_WORDS]
    else:
        tokens = [w for w in tokens if len(w) >= min_len]
    return tokens


def token_set(text: str, **kwargs) -> Set[str]:
    return set(tokenize(text, **kwargs))
