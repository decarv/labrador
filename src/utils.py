"""
utils

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
import collections
import datetime
import os
import pickle
import re
import time
import logging
from typing import Union, Optional

import psycopg2
import requests
import functools
import numpy as np
import pandas as pd
import torch
from psycopg2.extras import NamedTupleCursor, RealDictCursor, RealDictRow

import config
from models.metadata import Metadata
from config import DATA_DIR, POSTGRESQL_DB_NAME, POSTGRESQL_DB_USER, POSTGRESQL_DB_PASSWORD, \
    POSTGRESQL_DB_HOST, POSTGRESQL_DB_PORT


def log(func):
    """
    Decorator that creates an automatic logging for functions.

    Usage:
        @log
        def func(a, b):
            ...
    """
    logger = logging.getLogger(f"{__name__}.{func.__name__}")

    # noinspection PyMissingOrEmptyDocstring
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger.info(f"Running {func.__name__}")
            s = time.time()
            ret = func(*args, **kwargs)
            logger.info(f"{func.__name__} took {time.time() - s} seconds")
            return ret
        except Exception as e:
            logger.error(f"Exception raised in function {func.__name__}. Exception: {e}")
            raise e
    return wrapper


def get_request(
        url: str, timeout: int = 5, max_attempts: int = 5, wait_time: float = 10.0
) -> Optional[requests.Response]:
    """
    Abstracts HTTP GET request with requests library, using timeout, max_attempts and wait_time.

    :param url:
    :param timeout:
    :param max_attempts:
    :param wait_time:
    :return: Response or None if the request fails.
    """
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=timeout)
        except requests.exceptions.RequestException:
            return None
        if not response.ok:
            # logging.INFO(f"Response code was not 2XX: ({response.status_code=})")
            time.sleep(wait_time)
            continue
        return response

    # logging.warning(f"Failed to request the URI after {max_attempts} attempts")
    return None


def post_request(
        url: str, params: dict, headers: dict, data: dict,
        timeout: int = 5, max_attempts: int = 5, wait_time: float = 5.0
) -> Optional[requests.Response]:
    """
    Abstracts HTTP POST request with requests library, using params, headers, data (payload),
    timeout, max_attempts and wait_time. The function requests cookies before posting.

    :param url:
    :param params:
    :param headers:
    :param data:
    :param timeout:
    :param max_attempts:
    :param wait_time:
    :return: Response or None if the request fails.
    """
    cookies = get_request(url).cookies
    for attempt in range(max_attempts):
        # logging.INFO(f"Making POST request {attempt}/{max_attempts} to resource " + uri)
        try:
            response = requests.post(
                url, params=params, headers=headers, data=data, cookies=cookies, timeout=timeout
            )
        except requests.exceptions.RequestException:
            # logging.ERROR(f"Error making the request")
            return None

        if not response.ok:
            # logging.INFO(f"Response code was not 2XX: ({response.status_code=})")
            time.sleep(wait_time)
            continue
        return response

    # logging.ERROR(f"Failed to request the URI after {max_attempts} attempts")
    return None


def metadata_read() -> list[RealDictRow]:
    return db_read("metadata")


def tokenized_metadata_read() -> list[RealDictRow]:
    return db_read("tokenized_metadata")


def db_read(table_name: str) -> list[RealDictRow]:
    with psycopg2.connect(
        dbname=POSTGRESQL_DB_NAME,
        user=POSTGRESQL_DB_USER,
        password=POSTGRESQL_DB_PASSWORD,
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"SELECT * FROM {table_name};")
            instances = cursor.fetchall()
    return instances


@log
def tokenized_metadata_insert(
        metadata_url: str, 
        title_tokens_pt: list[str], 
        abstract_tokens_pt: list[str], 
        keywords_tokens_pt: list[str], 
        title_tokens_en: list[str], 
        abstract_tokens_en: list[str], 
        keywords_tokens_en: list[str]
) -> None:
    with psycopg2.connect(
        dbname=POSTGRESQL_DB_NAME,
        user=POSTGRESQL_DB_USER,
        password=POSTGRESQL_DB_PASSWORD,
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "INSERT INTO tokenized_metadata (metadata_url, title_tokens_pt, abstract_tokens_pt, keywords_tokens_pt, "
                "title_tokens_en, abstract_tokens_en, keywords_tokens_en) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s);",
                (metadata_url, title_tokens_pt, abstract_tokens_pt, keywords_tokens_pt, 
                 title_tokens_en, abstract_tokens_en, keywords_tokens_en)
            )
            conn.commit()


def embeddings_path(model_name: str, units_type: str, language: str, save_dir: str = config.EMBEDDINGS_DIR) -> str:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = f"{model_name}_{units_type}_{language}_embeddings.pt"
    path = os.path.join(save_dir, filename)
    return path


def embeddings_save(
        embeddings: torch.tensor, model_name: str, token_type: str, language: str, save_dir=config.EMBEDDINGS_DIR
) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_name = model_name.replace("/", "-")
    path = embeddings_path(model_name, token_type, language, save_dir)
    np.save(
        path,
        embeddings,
        allow_pickle=False
    )


def embeddings_load(model_name: str, units_type: str, language: str, save_dir=config.EMBEDDINGS_DIR) -> np.ndarray:
    model_name = model_name.replace("/", "-")
    path = embeddings_path(model_name, units_type, language, save_dir)
    return torch.load(path)


def indices_path(tokens_type: str, language: str, save_dir=config.INDICES_DIR) -> str:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = f"{tokens_type}_{language}_indices.pkl"
    path = os.path.join(save_dir, filename)
    return path


def indices_load(token_type: str, language: str, save_dir: str = config.INDICES_DIR) -> np.ndarray:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = indices_path(token_type, language, save_dir)
    ext = path.split(".")[-1]
    if ext == "pkl":
        return np.array(np.load(path, allow_pickle=True))
    elif ext == "npy":
        return np.load(path, allow_pickle=False)
    raise Exception("Invalid file extension.")


def indices_save(indices: list[int], token_type: str, language: str, save_dir: str = config.INDICES_DIR) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = indices_path(token_type, language, save_dir)
    with open(path, "wb") as f:
        pickle.dump(indices, f)


def load_metadata_from_csv() -> pd.DataFrame:
    metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"), keep_default_na=False)
    return metadata





def split_by_delimiters(string, delimiters: Union[list[str], str]) -> list[str]:
    """
    Splits the given string by one or multiple delimiters.

    Args:
        string: The string to be split
        delimiters: A string or list of strings used as delimiters

    Returns:
        A list of substrings.
    """
    if isinstance(delimiters, list):
        pattern = r"|".join(delimiters)
    else:
        pattern = r"{}".format(delimiters)
    return re.split(pattern, string)
