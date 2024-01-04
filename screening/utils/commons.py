
__all__ = ["set_logger", "set_task_logger", "create_folder", "sort_dict"]

import os, sys
import luigi, hashlib, six
from loguru import logger
from datetime import datetime


def set_logger(log_path, mode="w"):
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    log_level = os.environ.get('LOGURU_LEVEL', "INFO")
    logger.add(log_path, level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)


def set_task_logger(log_prefix, log_path):
    now = datetime.now()
    date = str(now.strftime("%d.%b.%Y"))
    log_name = log_prefix + "_%s.log" % date
    log_path = os.path.join(log_path, log_name)
    set_logger(log_path)


def create_folder(path):
    os.makedirs(path, exist_ok=True)


def sort_dict(item: dict):
    return {k: sort_dict(v) if isinstance(v, dict) else v for k, v in sorted(item.items())}



