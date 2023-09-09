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
import datetime
import os
import re
import time
import logging
from typing import Union, Optional

import psycopg2
import requests
import functools
import numpy as np
import pandas as pd

import config
from models.metadata import Metadata
from config import DATA_DIR, POSTGRESQL_DB_NAME, POSTGRESQL_DB_USER, POSTGRESQL_DB_PASSWORD, \
    POSTGRESQL_DB_HOST, POSTGRESQL_DB_PORT
import pickle


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


def load_metadata_from_db() -> list[Metadata]:
    conn = psycopg2.connect(
        dbname=POSTGRESQL_DB_NAME,
        user=POSTGRESQL_DB_USER,
        password=POSTGRESQL_DB_PASSWORD,
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT
    )
    res: list[Metadata] = []
    cursor = conn.cursor()
    cursor.execute("select * from metadata;")
    instances = cursor.fetchall()
    if instances is not None:
        for inst in instances:
            m = Metadata().parse_db_instance(inst)
            res.append(m)
    cursor.close()
    conn.close()
    return res


def embeddings_path(save_dir, model_name, units_type, language):
    filename = f"{model_name}_{units_type}_{language}_embeddings.npy"
    path = os.path.join(save_dir, filename)
    return path


def save_embeddings(embeddings, save_dir, model_name, units_type, language):
    model_name = model_name.replace("/", "-")
    path = embeddings_path(save_dir, model_name, units_type, language)
    np.save(
        path,
        embeddings,
        allow_pickle=False
    )


def load_embeddings(model_name: str, units_type: str, language: str, save_dir=config.EMBEDDINGS_DIR) -> np.ndarray:
    model_name = model_name.replace("/", "-")
    path = embeddings_path(save_dir, model_name, units_type, language)
    return np.load(path)


def indices_path(units_type: str, language: str, save_dir=config.INDICES_DIR) -> str:
    filename = f"{units_type}_{language}_indices.pkl"
    path = os.path.join(save_dir, filename)
    return path


def load_imap(units_type: str, language: str, save_dir=config.INDICES_DIR) -> np.ndarray:
    path = indices_path(units_type, language, save_dir)
    ext = path.split(".")[-1]
    if ext == "pkl":
        return np.array(np.load(path, allow_pickle=True))
    elif ext == "npy":
        return np.load(path, allow_pickle=False)
    raise Exception("Invalid file extension.")


def save_indices(indices, save_dir, units_type, language):
    path = indices_path(units_type, language, save_dir)

    with open(path, "wb") as f:
        pickle.dump(indices, f, pickle.HIGHEST_PROTOCOL)


def experiment_results_path(experiment_name, save_dir=config.EXPERIMENTS_RESULTS_DIR):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dt = datetime.datetime.now().strftime("%Y%m%d%H%M")
    filename = f"{dt}_{experiment_name}.json"
    path = os.path.join(save_dir, filename)
    return path


def load_metadata_from_csv() -> pd.DataFrame:
    metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"), keep_default_na=False)
    return metadata


def batch_generator(data, batch_size=64):
    """

    :param self:
    :param data:
    :param batch_size:
    :return:
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]


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
