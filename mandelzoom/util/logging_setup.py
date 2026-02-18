import logging
import logging.handlers
import multiprocessing as mp
from typing import Optional

_LOGGER_NAME = "mandelzoom"

def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)

def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s.%(msecs)03dZ %(processName)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

def configure_root_logging(
    *,
    level: int = logging.INFO,
    console: bool = True,
    log_file: Optional[str] = "render.log",
    rotate_bytes: int = 5 * 1024 * 1024,
    rotate_count: int = 5,
) -> logging.Logger:
    logger = get_logger()
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = _build_formatter()
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    if log_file:
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=rotate_bytes, backupCount=rotate_count, encoding="utf-8"
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def create_log_queue() -> mp.Queue:
    return mp.Queue(-1)

def start_queue_listener(queue: mp.Queue, listener_logger: logging.Logger) -> logging.handlers.QueueListener:
    handlers = list(listener_logger.handlers)
    listener = logging.handlers.QueueListener(queue, *handlers, respect_handler_level=True)
    listener.start()
    return listener

def configure_worker_logging(queue: mp.Queue, *, level: int = logging.INFO) -> None:
    logger = get_logger()
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    qh = logging.handlers.QueueHandler(queue)
    qh.setLevel(level)
    logger.addHandler(qh)

def logging_initialiser(queue: mp.Queue, level: int) -> None:
    configure_worker_logging(queue, level=level)
