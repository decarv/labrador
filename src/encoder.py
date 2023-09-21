"""
encoder

Copyright 2023 Henrique Araújo de Carvalho

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
import sys
from typing import Iterator, Union, Optional
from torch.cuda import OutOfMemoryError

import nvidia_smi

import torch
from sentence_transformers import SentenceTransformer

import os
import config
import utils
from utils import log

logger = utils.configure_logger(__name__)
log = log(logger)


class Encoder:

    def __init__(self, model_name: str, token_type: str, language: str = "pt", cache: bool = True):
        self.cache = cache
        self.device = 'cuda:0'
        self.language = language
        self.token_type = token_type
        self.model_name = model_name.split("/")[-1]
        self.model = SentenceTransformer(
            model_name, device=self.device, cache_folder=config.MODEL_CACHE_DIR
        )

        nvidia_smi.nvmlInit()
        gpu_index: int = 0
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_index)
        self.chunk_number: int = 0

    # TODO: improve embeddings_save to store indices
    def encode(self, chunk_iterator: Iterator[list[list[str]]]):
        for chunk in chunk_iterator:
            path = utils.embeddings_path(self.model_name, self.token_type, self.language, self.chunk_number)
            if config.ALLOW_CACHING and os.path.exists(path):
                logger.info(f"Embeddings already cached: {self.chunk_number}. Skipping...")
            else:
                embeddings: torch.tensor = self._encode_chunk(chunk)
                logger.info("Saving embeddings...")
                if len(embeddings) == 0:
                    logger.error(f"Embeddings of {self.chunk_number} are EMPTY. Skipping...")
                else:
                    utils.embeddings_save(embeddings, self.model_name, self.token_type, self.language, self.chunk_number)
                    logger.info("Embeddings saved.")
            self.chunk_number += 1

    @log
    def _get_uniform_sized_batches(self, chunk: list[str]) -> torch.tensor:
        percent_of_gpu_mem = 2500  # at most 3GiB of the GPU memory
        max_estimated_mem = percent_of_gpu_mem * 80  # between 65 and 80% of the GPU memory

        i: int = 0
        batches: list[list[str]] = [[]]
        batch_size: int = 0
        avg_batch_size: int = 0
        for volume in chunk:
            size_of_volume: int = len(volume.encode('utf-8'))  # sys.getsizeof(volume)
            if batch_size + size_of_volume < max_estimated_mem:
                batches[i].append(volume)
                batch_size += size_of_volume
            else:
                i += 1
                batches.append([volume])
                batch_size = size_of_volume
            avg_batch_size += size_of_volume
        avg_batch_size = avg_batch_size // len(batches)
        avg_len_batch = sum([len(batch) for batch in batches]) // len(batches)
        logger.info("Attempt to create uniform sized batches:")
        for i, batch in enumerate(batches):
            logger.info(f"Batch {i} has {len(batch)} items and {len(''.join(batch))} bytes")
        logger.info(f"{len(batches)} batches of {avg_batch_size} mb on average and {avg_len_batch} items on average")
        return batches

    @log
    def _encode_batches(self, batches: list[list[str]]) -> torch.tensor:
        embeddings: list[torch.tensor] = []
        for batch in batches:
            with torch.no_grad():
                torch.cuda.empty_cache()
                batch_embeddings: torch.tensor = self.model.encode(batch, batch_size=len(batch), convert_to_tensor=True)
                embeddings.append(batch_embeddings.cpu())
        return torch.cat(embeddings)

    @log
    def _encode_chunk_2(self, chunk: list[list[str]]) -> torch.tensor:
        chunk: list[str] = self._flatten(chunk)
        batches = self._get_uniform_sized_batches(chunk)
        embeddings: torch.tensor = self._encode_batches(batches)
        return embeddings

    @log
    def _encode_chunk(self, chunk: list[list[str]]) -> torch.tensor:
        chunk: list[str] = self._flatten(chunk)
        batch_size = 256  # self._get_batch_size(chunk)
        while batch_size > 0:
            try:
                return torch.cat(self._encode_batch(chunk, batch_size))
            except OutOfMemoryError:
                batch_size //= 2
                logger.error(f"OutOfMemoryError. Reducing batch size to {batch_size}.")
            except RuntimeError as e:
                logger.exception(f"Ex: {e} in chunk {self.chunk_number}")
                return torch.tensor([])
        logger.exception("OutOfMemoryError. Batch size is 1. Skipping chunk.")
        return torch.tensor([])

    def _encode_batch(self, batch: list[str], batch_size=64) -> list[torch.tensor]:
        embeddings: list[torch.tensor] = []
        for batch in self._batch_generator(batch, batch_size=batch_size):
            with torch.no_grad():
                torch.cuda.empty_cache()
                batch_embeddings: torch.tensor = self.model.encode(
                    batch, batch_size=len(batch), convert_to_tensor=True
                )
                embeddings.append(batch_embeddings.cpu())
        return embeddings

    @log
    def _get_batch_size(self, chunk: list[str]) -> int:
        batch_size = 256
        estimated_mem_overhead = 0.7
        free_mem_mb = self._get_gpu_mem_info()
        while batch_size > 0:
            for batch in self._batch_generator(chunk, batch_size=batch_size):
                if sys.getsizeof("".join(batch)) > free_mem_mb * estimated_mem_overhead:
                    batch_size = int(batch_size * 0.8)
                    break
            return batch_size
        return batch_size

    def _get_gpu_mem_info(self):
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        free_mem_mb = info.free
        return free_mem_mb

    @staticmethod
    def _flatten(data: list[list[str]]) -> list[str]:
        return [item for sublist in data for item in sublist]

    @staticmethod
    def _flatten_nested(ds):
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

    @staticmethod
    def _batch_generator(data: list[str], batch_size: int = 32):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
