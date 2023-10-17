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

import os
import pickle
import time
from typing import Optional, Any
import requests
import pandas as pd
import torch

import config
from config import DATA_DIR
import httpx


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

async def fetch_async(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response

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


def embeddings_path(
        model_name: str, token_type: str, language: str, chunk_number: int = -1, save_dir: str = config.EMBEDDINGS_DIR
) -> str:
    directory = os.path.join(save_dir, model_name)
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{language}_{token_type}_embeddings")

    if int(chunk_number) >= 0:
        filepath += f"_{chunk_number}.pt"
    else:
        filepath += ".pt"
    return filepath


def embeddings_save(
        embeddings: torch.tensor, model_name: str, token_type: str, language: str, chunk_number: int = -1,
        save_dir=config.EMBEDDINGS_DIR
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    model_name = model_name.replace("/", "-")
    path = embeddings_path(model_name, token_type, language, chunk_number)
    torch.save(embeddings, path)


def indices_load(token_type: str, language: str, save_dir: str = config.INDICES_DIR) -> tuple[list[int], list[int]]:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(indices_path(token_type, language, save_dir), "rb") as f:
        loaded_data = pickle.load(f)
        return loaded_data[0], loaded_data[1]


def indices_path(
        tokens_type: str, language: str, save_dir: str = config.INDICES_DIR
) -> str:
    directory = os.path.join(save_dir)
    os.makedirs(directory, exist_ok=True)
    filename = f"{language}_{tokens_type}_indices.pkl"
    path = os.path.join(directory, filename)
    return path


def indices_save(
        indices: tuple[list[int], list[int]], token_type: str, language: str, save_dir: str = config.INDICES_DIR
) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = indices_path(token_type, language, save_dir)
    with open(path, "wb") as f:
        pickle.dump(indices, f)


def tokens_path(
        tokens_type: str, language: str, save_dir: str = config.TOKENS_DIR
) -> str:
    directory = os.path.join(save_dir)
    os.makedirs(directory, exist_ok=True)
    filename = f"{language}_{tokens_type}_tokens.pkl"
    path = os.path.join(directory, filename)
    return path


def tokens_save(tokens: list[list[str]], token_type: str, language: str, save_dir: str = config.TOKENS_DIR) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = tokens_path(token_type, language, save_dir)
    with open(path, "wb") as f:
        pickle.dump(tokens, f)


def tokens_load(token_type: str, language: str, save_dir: str = config.TOKENS_DIR) -> list[list[str]]:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(tokens_path(token_type, language, save_dir), "rb") as f:
        return pickle.load(f)


def load_metadata_from_csv() -> pd.DataFrame:
    metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"), keep_default_na=False)
    return metadata


def flatten(data: list[list[Any]]) -> list[Any]:
    return [item for sublist in data for item in sublist]


def flatten_nested(ds):
    flat_ds = []
    stack = [ds[i] for i in range(len(ds) - 1, -1, -1)]
    while len(stack) > 0:
        node = stack.pop()
        if isinstance(node, str):
            flat_ds.append(node)
        else:
            for i in range(len(node) - 1, -1, -1):
                stack.append(node[i])
    return flat_ds


def collection_name(model_name, token_type):
    return f"{model_name.replace('-', '').replace('/', '')}{token_type.replace('_', '')}"
