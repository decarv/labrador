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
from typing import Iterator

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
        self.model.to(self.device)

        nvidia_smi.nvmlInit()
        gpu_index: int = 0
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_index)
        self.chunk_number: int = 0

    def encode(self, chunk_iterator: Iterator[list[str]]):
        for chunk in chunk_iterator:
            if config.ALLOW_CACHING and os.path.exists(utils.embeddings_path(self.model_name, self.token_type, self.language, self.chunk_number)):
                logger.info(f"Embeddings already cached: {self.chunk_number}. Skipping...")
            else:
                embeddings: torch.tensor = self._encode_chunk(chunk)
                logger.info("Saving embeddings...")
                utils.embeddings_save(embeddings, self.model_name, self.token_type, self.language, self.chunk_number)
                logger.info("Embeddings saved.")
            self.chunk_number += 1

    @log(logger)
    def _encode_chunk(self, batch: list[str]) -> torch.tensor:
        embeddings: list[torch.tensor] = []
        for batch in self.batch_generator(batch):
            
            with torch.no_grad():
                try:
                    batch_embeddings: torch.tensor = self.model.encode(
                        batch, batch_size=len(batch), convert_to_tensor=True
                    )
                    embeddings.append(batch_embeddings.cpu())
                    free_mem_mb = self.get_gpu_mem_info()
                    if free_mem_mb < 200:
                        logger.info(f"Free memory: {free_mem_mb} MB. Cleaning cache.")
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.exception(f"Ex: {e}")
        return torch.cat(embeddings)

    def get_gpu_mem_info(self):
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        free_mem_mb = info.free / (1024**2)
        return free_mem_mb

    def batch_generator(self, data: list[str]):
        batch_size = 64
        # free_mem_mb = self.get_gpu_mem_info()
        # data_size_mb = sys.getsizeof(data) / (1024**2)
        # batch_size: int = int(max((free_mem_mb // data_size_mb) * 0.8, batch_size))
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

