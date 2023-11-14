"""
log

Copyright 2023 Henrique de Carvalho

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import logging
import os
import time
from logging import handlers

from labrador import config

logger = None


def configure_logger(path):
    global logger
    # Get log name and filename
    full_path = os.path.abspath(path)
    filename = os.path.basename(full_path)
    name = os.path.splitext(filename)[0]
    log_filename: str = f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(config.LOG_LEVEL)
        console_formatter = logging.Formatter(config.LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Set up file handler
        log_formatter = logging.Formatter(config.LOG_FORMAT)
        logfile_handler = handlers.RotatingFileHandler(
                os.path.join(config.LOG_DIR, log_filename),
                maxBytes=5 * 1024 * 1024,
                backupCount=10
        )
        logfile_handler.setFormatter(log_formatter)
        logger.addHandler(logfile_handler)

    return logger


def log(func):
    """
    Decorator for automatic logging.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if logger is None:
                raise Exception("LOGGER is not configured. Call configure_logger() before using @log.")
            try:
                logger.debug(f"Running {func.__name__}")
                s = time.time()
                ret = func(*args, **kwargs)
                logger.debug(f"{func.__name__} took {time.time() - s} seconds")
                return ret
            except Exception as e:
                logger.error(f"Exception raised in function {func.__name__}. Exception: {type(e).__name__} - {e}")
                raise e
        return wrapper
    return decorator
