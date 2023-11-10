"""
indexer

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

import time
from multiprocessing import Pool
from typing import Iterator
import httpcore
import httpx
import qdrant_client.http.exceptions

import config
from tokenizer import Tokenizer
from util.log import configure_logger
from util.database import Database
from util.utils import collection_name

from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

BATCH_SIZE: int = 16

logger = configure_logger(__file__)


class SparseIndexer:
    pass


class NeuralIndexer:
    def __init__(self, client: QdrantClient, database: Database, model_name: str, token_type: str, id_range: tuple[int, int] = None):
        self._index_client = client
        self._database = database
        self.collection_name = collection_name(model_name, token_type)
        self.model_name = model_name
        self.token_type = token_type

    def create_collection(self):
        if self.collection_name in [c.name for c in self._index_client.get_collections().collections]:
            logger.error(f"Collection {self.collection_name} already exists.")
            return

        try:
            self._index_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=config.VECTORS_DIMENSION,
                    distance=models.Distance.COSINE if config.MODELS[self.model_name] == 'cosine' else models.Distance.DOT,
                    on_disk=True,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
                shard_number=2,
                timeout=60,
            )
        except Exception as e:
            logger.warning(f"Collection already exists: {e}")
            pass
        logger.info("Collection created.")

    def collection_exists(self):
        return self.collection_name in [c.name for c in self._index_client.get_collections().collections]

    def run(self, id_range: tuple[int, int] = None):
        if not self.collection_exists():
            self.create_collection()

        if id_range is None:
            id_range = (1, self._database.select("""SELECT id FROM embeddings ORDER BY id DESC LIMIT 1;""")[0]['id'])

        batches: Iterator[list[dict]] = self._database.batch_generator(
            """SELECT d.id as doc_id, d.title_pt as title,
        d.abstract_pt as abstract, d.author,
        d.keywords_pt as keywords, e.vector, t.token, t.id as token_id, e.id as embeddings_id
           FROM embeddings AS e
           JOIN tokens AS t ON e.token_id = t.id
           JOIN documents AS d ON t.doc_id = d.id
           LEFT JOIN indexed AS i ON e.id = i.embeddings_id
          WHERE e.model_name = %s
            AND t.token_type = %s
            AND i.id IS NULL
            AND e.id BETWEEN %s AND %s;
            """,
            (self.model_name, self.token_type, *id_range)
        )

        for batch in batches:
            self.insert_batch(batch)

    def insert_batch(self, batch: list[dict]):
        points: list[PointStruct] = []
        for record in batch:
            points.append(
                PointStruct(
                    vector=record['vector'],
                    payload={
                         "doc_id": record['doc_id'],
                         "title": record['title'],
                         "abstract": record['abstract'],
                         "keywords": record['keywords'],
                         "author": record['author'],
                         "token": record['token'],
                            },
                    id=record['token_id']
                )
            )
        try:
            self.insert(points)
            self._database.insert_many("""INSERT INTO indexed (embeddings_id) VALUES (%s)""",
                                        [(record['embeddings_id'],) for record in batch])
        except Exception as e:
            logger.error(f"NeuralIndex : {type(e)} : {e}")
            self._database.insert_error(f"NeuralIndex : {type(e)} : {e}")

    def insert(self, points: list[PointStruct]):
        try:
            self._index_client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        except (TimeoutError,
                qdrant_client.http.exceptions.ResponseHandlingException, httpcore.ReadTimeout, httpx.ReadTimeout):
            for point in points:
                try:
                    self._index_client.upsert(
                        collection_name=self.collection_name,
                        points=[point],
                    )
                except (TimeoutError,
                        qdrant_client.http.exceptions.ResponseHandlingException, httpcore.ReadTimeout,
                        httpx.ReadTimeout):
                    logger.error(f"Erro ao inserir ponto: {point}.")
                    self._database.insert_error(f"Erro ao inserir ponto: {point}.")


def get_id_ranges(num_workers, model_name, token_type):
    database: Database = Database()
    ids = [row['embedding_id'] for row in database.select(
        """SELECT e.id AS embedding_id
           FROM embeddings AS e
           JOIN tokens AS t ON e.token_id = t.id
           LEFT JOIN indexed AS i ON e.id = i.embeddings_id
          WHERE e.model_name = %s
            AND t.token_type = %s
            AND i.id IS NULL
          ORDER BY e.id;
        """, (model_name, token_type,))]

    if not ids:
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


def run_parallel(params):
    model_name, token_type, id_range = params
    database: Database = Database()
    index_client = QdrantClient(config.QDRANT_HOST, port=6333)
    indexer: NeuralIndexer = NeuralIndexer(index_client, database, model_name, token_type)
    indexer.run(id_range)


if __name__ == "__main__":
    num_indexer_workers = 2
    while True:
        for model_name in list(config.MODELS.keys()):
            for token_type in Tokenizer.token_types():
                id_ranges: list[tuple[int, int]] = get_id_ranges(num_indexer_workers, model_name, token_type)
                params = [(model_name, token_type, id_range) for id_range in id_ranges]
                with Pool(processes=num_indexer_workers) as pool:
                    pool.map(run_parallel, params)
