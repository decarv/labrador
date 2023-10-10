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
import torch
from sentence_transformers import SentenceTransformer
from typing import Iterator
from psycopg2.extras import RealDictRow
from torch.cuda import OutOfMemoryError
import nvidia_smi

import labrador.config
from util import database, log

logger = log.configure_logger(__file__)
log = log.log(logger)


class Encoder:
    @log
    def __init__(self, model_name: str, token_type: str, language: str = "pt", cache: bool = True):
        logger.debug("Initializing encoder...")

        self.cache = cache
        self.device = 'cuda:0'
        self.language = language
        self.token_type = token_type
        self.model_name = model_name.split("/")[-1]
        self.model = SentenceTransformer(model_name, device=self.device, cache_folder=config.MODEL_CACHE_DIR)

        nvidia_smi.nvmlInit()
        gpu_index: int = 0
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_index)

        logger.debug("Done initializing encoder.")

    @log
    def encode(self, chunk_generator: Iterator[list[RealDictRow]]):
        for chunk in chunk_generator:
            tokens_ids: list[int] = []
            tokens: list[str] = []
            for row in chunk:
                tokens_ids.append(row['id'])
                tokens.append(row['token'])

            batch_size: int = 64  # self._get_batch_size(chunk)
            batch_encoded: bool = False
            while batch_size > 0 and not batch_encoded:
                generator_length: int = len(chunk) // batch_size
                generator: Iterator[tuple[list[int], list[str]]] = self._batch_generator(tokens_ids, tokens, batch_size)
                for i, (batch_tokens_ids, batch) in enumerate(generator):
                    try:
                        logger.debug(f"Encoding batch {i} of {generator_length}")
                        logger.debug("Filtering already encoded tokens...")
                        not_encoded_tokens_ids: list[int] = []
                        not_encoded_tokens: list[str] = []
                        for j, token_id in enumerate(batch_tokens_ids):
                            if not database.embeddings_exists(token_id, self.model_name):
                                not_encoded_tokens_ids.append(token_id)
                                not_encoded_tokens.append(batch[j])
                        batch = not_encoded_tokens
                        batch_tokens_ids = not_encoded_tokens_ids
                        logger.debug(f"Batch filtered. Length is now {len(batch)}")

                        if len(batch) == 0:
                            logger.debug(f"Batch {i} is empty. Skipping...")
                            continue
                        embeddings: torch.tensor = self._encode_batch(batch)
                        database.embeddings_insert(batch_tokens_ids, embeddings, self.model_name)

                    except OutOfMemoryError as e:
                        logger.debug(f"Failed at token_id: {batch_tokens_ids[0]}")
                        tokens_ids = tokens_ids[i * batch_size:]
                        tokens = tokens[i * batch_size:]
                        batch_size //= 2
                        logger.debug(f"{e}: Reducing batch size to {batch_size}")
                        logger.debug(f"Restarting encoding process at token_id: {tokens_ids[0]}")
                        break

                    except Exception as e:
                        error_message: str = (
                            f"Ex: {e} | "
                            f"batch: {batch} | "
                            f"batch_size {batch_size} | "
                            f"ids: {batch_tokens_ids}"
                        )
                        logger.exception(error_message)
                        database.errors_insert(error_message)
                batch_encoded = True

    @log
    def _encode_batch(self, batch: list[str]) -> torch.tensor:
        with torch.no_grad():
            torch.cuda.empty_cache()
            embeddings: torch.tensor = self.model.encode(
                batch, batch_size=len(batch), convert_to_tensor=True
            )
            return embeddings

    @staticmethod
    def _batch_generator(ids: list[int], data: list[str], batch_size: int) -> Iterator[tuple[list[int], list[str]]]:
        for i in range(0, len(data), batch_size):
            yield ids[i:i + batch_size], data[i:i + batch_size]

    @staticmethod
    def _calculate_uniform_sized_batches(chunk: list[str]) -> torch.tensor:
        percent_of_gpu_mem = 2500  # at most 3GiB of the GPU memory
        max_estimated_mem = percent_of_gpu_mem * 80  # between 65 and 80% of the GPU memory

        i: int = 0
        batches: list[list[str]] = [[]]
        batch_size: int = 0
        avg_batch_size: int = 0
        for volume in chunk:
            size_of_volume: int = len(volume.encode('utf-8'))
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

    def _get_gpu_mem_info(self):
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        free_mem_mb = info.free
        return free_mem_mb


if __name__ == "__main__":
    chunk_size: int = 1024 * 4
    languages = ["pt"]
    token_types = ["sentence_with_keywords"]  # TODO: replace for Tokenizer.token_types()
    for language in languages:
        logger.info(f"main: Language {language}")
        for token_type in token_types:
            logger.info(f"main: Token Type: {token_type}")
            for model in config.MODELS:
                logger.info(f"main: Model: {model}")
                encoder: Encoder = Encoder(model_name=model, token_type=token_type)
                tokens_generator: Iterator[list[RealDictRow]] = database.tokens_chunk_generator(encoder.model_name)
                encoder.encode(tokens_generator)
