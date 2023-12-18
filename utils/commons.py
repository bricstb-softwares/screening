import os
import logging
from datetime import datetime


def set_logger(log_path, mode="w"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, mode=mode)
    formatter = logging.Formatter("%(asctime)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ColoredFormatter())
    logger.addHandler(stream_handler)


def set_task_logger(log_prefix, log_path):
    now = datetime.now()
    date = str(now.strftime("%d.%b.%Y"))
    log_name = log_prefix + "_%s.log" % date
    log_path = os.path.join(log_path, log_name)
    set_logger(log_path)


class ColoredFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors."""

    GREY = "\x1b[90;21m"
    YELLOW = "\x1b[33;21m"
    RED = "\x1b[31;21m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    FORMAT = "%(asctime)s %(message)s"
    FORMAT2 = "%(asctime)s %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: GREY + FORMAT2 + RESET,
        logging.INFO: GREY + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT2 + RESET,
        logging.ERROR: RED + FORMAT2 + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT2 + RESET,
    }

    def format(self, record: logging.LogRecord):
        """Overwrite parent's format."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%y-%m-%dT%H:%M:%S")
        return formatter.format(record)


def create_folder(path):
    os.makedirs(path, exist_ok=True)
