import logging
import os
import sys

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    return logger


def set_level(level: str) -> None:
    """设置全局日志级别，如 'DEBUG', 'INFO', 'WARNING', 'ERROR'"""
    global _LOG_LEVEL
    _LOG_LEVEL = level.upper()
    # 更新所有已创建的logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
