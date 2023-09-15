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

    def encode(self, chunk_iterator: Iterator[list[str]]):
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

    @log(logger)
    def _encode_chunk(self, chunk: list[str]) -> torch.tensor:
        batch_size = 128
        embeddings: list[torch.tensor] = []
        while batch_size > 1:
            try:
                embeddings = self._encode_batch(chunk, batch_size)
                concat_embeddings: torch.tensor = torch.cat(embeddings)
                return concat_embeddings
            except OutOfMemoryError:
                batch_size //= 2
                logger.info(f"OutOfMemoryError. Reducing batch size to {batch_size}.")
            except RuntimeError as e:
                logger.exception(f"Ex: {e} in chunk {self.chunk_number}")
                return torch.tensor([])

        logger.exception("OutOfMemoryError. Batch size is 1. Skipping chunk.")
        return torch.tensor([])

    def _encode_batch(self, batch: list[str], batch_size=64) -> list[torch.tensor]:
        embeddings: list[torch.tensor] = []
        for batch in self.batch_processor(batch, batch_size=64):
            with torch.no_grad():
                torch.cuda.empty_cache()
                batch_embeddings: torch.tensor = self.model.encode(
                    batch, batch_size=len(batch), convert_to_tensor=True
                )
                embeddings.append(batch_embeddings.cpu())
        return embeddings

    def get_gpu_mem_info(self):
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        free_mem_mb = info.free / (1024**2)
        return free_mem_mb

    def _flatten(self, data: Union[list[str], list[list[str]]]) -> list[str]:
        if isinstance(data[0], list):
            tmp_batch = []
            for sublist in data:
                tmp_batch.extend(sublist)
            return tmp_batch
        else:
            return data

    def batch_processor(self, data: list[str], batch_size: int = 256):
        flattened_batch: list[str] = self._flatten(data)
        for i in range(0, len(flattened_batch), batch_size):
            yield flattened_batch[i:i + batch_size]
