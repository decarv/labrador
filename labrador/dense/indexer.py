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

import httpx
import httpcore
from typing import Iterator, Any
from multiprocessing import Pool
from qdrant_client import QdrantClient, models
import qdrant_client.http.exceptions
from qdrant_client.models import PointStruct

from labrador import config
from labrador.dense.tokenizer import Tokenizer
from labrador.util.log import configure_logger
from labrador.util.database import Database

from labrador import util

BATCH_SIZE: int = 16

logger = configure_logger(__file__)


class DenseIndexer:
    def __init__(self, client: QdrantClient, database: Database, model_name: str, token_type: str, workers: int = 1):
        self._index_client = client
        self._database = database
        self._workers = workers

        self.model_name = model_name
        self.token_type = token_type
        self.collection_name = util.collection_name(model_name, token_type)

    def run(self, id_range: tuple[int, int] = None):
        if not self._collection_exists():
            self._create_collection()

        if id_range is None:
            id_range = (1, self._database.select("""SELECT id FROM embeddings ORDER BY id DESC LIMIT 1;""")[0]['id'])

        batches: Iterator[list[dict]] = self._database.batch_generator(
            """
             SELECT d.id as doc_id, d.title_pt as title,
                    d.abstract_pt as abstract, d.author,
                    d.keywords_pt as keywords, e.vector, t.token, t.id as token_id, e.id as embeddings_id
               FROM embeddings AS e
               JOIN tokens AS t ON e.token_id = t.id
               JOIN documents AS d ON t.doc_id = d.id
               LEFT JOIN indexed AS i ON e.id = i.embeddings_id
              WHERE e.model_name = %s
                AND t.token_type = %s
                AND i.id IS NULL
                AND e.id BETWEEN %s AND %s;""",
            (self.model_name, self.token_type, *id_range)
        )

        for batch in batches:
            self.insert_records(batch)

    def insert_records(self, batch: list[dict]) -> None:
        """
        Transforms a batch of records into a list of PointStructs and inserts them into the index.
        :param batch: list of records to be inserted into the index.
        :return: None
        """
        points: list[PointStruct] = []
        for record in batch:
            points.append(self._create_point_from_record(record))
        try:
            self.insert_points(points)
        except Exception as e:
            logger.error(f"NeuralIndex : {type(e)} : {e}")
            self._database.insert_error(f"NeuralIndex : {type(e)} : {e}")

    def insert_points(self, points: list[PointStruct]) -> None:
        try:
            self._index_client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            self._database.insert_many("""INSERT INTO indexed (embeddings_id) VALUES (%s)""",
                                       [(point.embeddings_id,) for point in points])
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

    def _create_collection(self) -> None:
        """
        Wrapper for the create_collection method of the QdrantClient. It checks if the collection already exists before
        creating it.
        :return: None
        """
        if self.collection_name in [c.name for c in self._index_client.get_collections().collections]:
            logger.error(f"Collection {self.collection_name} already exists.")
            return

        try:
            distance: models.Distance = models.Distance.DOT
            if config.MODELS[self.model_name] == 'cosine':
                distance = models.Distance.COSINE
            self._index_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=config.VECTORS_DIMENSION,
                    distance=distance,
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

    def _collection_exists(self):
        """
        Checks if the collection already exists.
        """
        return self.collection_name in [c.name for c in self._index_client.get_collections().collections]

    def run_parallel(self):
        workers, id_ranges = self._distribute_workload()
        with Pool(processes=self._workers) as pool:
            pool.map(self.run, id_ranges)

    def _distribute_workload(self) -> tuple[int, list[tuple[int, int]]]:
        ids = [row['embedding_id'] for row in self._database.select(
            """SELECT e.id AS embedding_id
               FROM embeddings AS e
               JOIN tokens AS t ON e.token_id = t.id
               LEFT JOIN indexed AS i ON e.id = i.embeddings_id
              WHERE e.model_name = %s
                AND t.token_type = %s
                AND i.id IS NULL
              ORDER BY e.id;
            """, (self.model_name, self.token_type,))]

        return util.distribute_workload(ids, self._workers)

    @staticmethod
    def _create_point_from_record(record: dict[str, Any]) -> PointStruct:
        return PointStruct(
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


if __name__ == "__main__":
    database = Database()
    index_client = QdrantClient(config.QDRANT_HOST, port=6333)
    for model_name in list(config.MODELS.keys()):
        for token_type in Tokenizer.token_types():
            indexer = DenseIndexer(database=database, client=index_client, model_name=model_name, token_type=token_type,
                                   workers=2)
            indexer.run_parallel()
