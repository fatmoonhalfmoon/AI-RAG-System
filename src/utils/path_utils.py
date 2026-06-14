import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_path(path: str, must_exist: bool = False) -> str:
    """校验路径安全：必须在项目目录内，禁止路径遍历"""
    if not path or not isinstance(path, str):
        raise ValueError("路径不能为空")
    normalized = os.path.normpath(path)
    parts = normalized.replace("\\", "/").split("/")
    if ".." in parts:
        raise ValueError(f"路径不允许包含 '..': {path}")
    abs_path = os.path.abspath(normalized)
    abs_root = os.path.abspath(PROJECT_ROOT)
    if not (abs_path.startswith(abs_root + os.sep) or abs_path == abs_root):
        raise ValueError(f"路径必须在项目目录内: {path} (项目根: {abs_root})")
    if must_exist and not os.path.exists(abs_path):
        raise FileNotFoundError(f"路径不存在: {abs_path}")
    return abs_path
