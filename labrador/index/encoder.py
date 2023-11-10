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
import time
from multiprocessing import Pool

import torch
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from typing import Iterator
from psycopg2.extras import RealDictRow
from torch.cuda import OutOfMemoryError
import nvidia_smi

import config
from tokenizer import Tokenizer
from util.database import Database
from util.log import configure_logger

logger = configure_logger(__file__)


class Encoder:
    """
    TODO: Ideally, encoder should not have a token_type. It should fetch from the database and encode all tokens.
        But for now, it is only encoding sentence_with_keywords tokens.
    """
    def __init__(self, database: Database, model_name: str, token_type: str = 'sentence_with_keywords'):
        self._database = database
        self.model_name = model_name
        self.token_type = token_type

        self.model = SentenceTransformer(model_name, device='cuda:0', cache_folder=config.MODEL_CACHE_DIR)

        nvidia_smi.nvmlInit()
        gpu_index: int = 0
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_index)

    def run(self, id_range: tuple[int, int] = None):
        if id_range is None:
            id_range = (1, self._database.select("""SELECT id FROM tokens ORDER BY id DESC LIMIT 1;""")[0]['id'])

        tokens_batches: Iterator[list] = self._database.batch_generator(
            """SELECT t.id, t.token, t.token_type
               FROM tokens AS t
               WHERE t.id not in (SELECT token_id FROM embeddings WHERE model_name = %s)
               AND t.token_type = %s
               AND t.id BETWEEN %s AND %s
            """,
            (self.model_name, self.token_type, *id_range)
        )
        self.encode(tokens_batches)

    def encode(self, batches: Iterator[list]):
        for batch in batches:

            # minibatch_size is reduced by half until reaching 0, or encoding of whole batch is successful,
            # if the encoding process fails due to OutOfMemoryError.
            first: int = 0
            batch_encoded: bool = False
            minibatch_size: int = 16  # self._get_batch_size(chunk)
            while minibatch_size > 0 and not batch_encoded:
                generator: Iterator[list[dict]] = self._minibatch_generator(batch, first, minibatch_size)
                for i, minibatch in enumerate(generator):
                    try:
                        minibatch_tokens: list[str] = [row['token'] for row in minibatch]
                        minibatch_tokens_ids: list[int] = [row['id'] for row in minibatch]

                        embeddings: torch.tensor = self._encode_minibatch(minibatch_tokens)

                        self._database.insert_many(
                            """INSERT INTO embeddings (token_id, vector, model_name) VALUES (%s, %s, %s)""",
                            [
                                (token_id, vector.tolist(), self.model_name)
                                for token_id, vector in zip(minibatch_tokens_ids, embeddings)
                            ]
                        )

                        first = (i + 1) * minibatch_size

                    except OutOfMemoryError as e:
                        minibatch_size //= 2
                        logger.warning(f"{e}: Reducing batch size to {minibatch_size}")
                        break

                    except Exception as e:
                        error_message: str = f"Encoder : {type(e)} : {e}"
                        logger.exception(error_message)
                        self._database.insert_error(error_message)

                batch_encoded = True

    def _encode_minibatch(self, batch: list[str]) -> torch.tensor:
        with torch.no_grad():
            torch.cuda.empty_cache()
            embeddings: torch.tensor = self.model.encode(
                batch, batch_size=len(batch), convert_to_tensor=True
            )
            return embeddings

    @staticmethod
    def _minibatch_generator(batch: list[dict], first: int, minibatch_size: int) -> Iterator[list[dict]]:
        """Creates a minibatch generator from a batch of tokens."""
        for i in range(first, len(batch), minibatch_size):
            yield batch[i:i + minibatch_size]

    @staticmethod
    def _minibatch_generator_depr(ids: list[int], data: list[str], batch_size: int) -> Iterator[tuple[list[int], list[str]]]:
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


def get_id_ranges(db, num_workers, model_name, token_type):
    ids = [row['id'] for row in db.select(
        f"""SELECT t.id 
                  FROM tokens AS t
                  WHERE t.id not in (SELECT token_id FROM embeddings WHERE model_name = %s) 
                  AND token_type = %s ORDER BY id;""", (model_name, token_type,))]

    if not ids:
        logger.info(f"No tokens found for token_type {token_type}.")
        return []

    num_workers = min(len(ids), num_workers)

    id_ranges = []
    range_length = len(ids) // num_workers
    first: int = 0
    for i in range(num_workers):
        last = range_length * (i + 1)
        if last >= len(ids):
            last = -1
        id_ranges.append((ids[first], ids[last]))
        first = last + 1
    return id_ranges


if __name__ == "__main__":
    database: Database = Database()
    num_encoder_workers: int = 2
    while True:
        for model_name in config.MODELS.keys():
            for token_type in Tokenizer.token_types():
                encoder: Encoder = Encoder(database, model_name)
                id_ranges = get_id_ranges(database, num_encoder_workers, model_name, token_type)
                if not id_ranges:
                    continue
                with Pool(processes=num_encoder_workers) as pool:
                    pool.map(encoder.run, id_ranges)
