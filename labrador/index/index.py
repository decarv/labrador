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
from typing import Iterator
import httpcore
import httpx
import qdrant_client.http.exceptions

import config
from util.log import configure_logger
from util.database import Database
from util.utils import collection_name

from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

BATCH_SIZE: int = 16

logger = configure_logger(__file__)


class SparseIndexer:
    pass


class NeuralIndex:
    def __init__(self, database: Database, model_name: str, token_type: str, vectorizer: str = 'none'):
        self.client = QdrantClient(config.QDRANT_HOST, port=6333)
        self.collection_name = collection_name(model_name, token_type)
        self.model_name = model_name
        self.token_type = token_type

        self._database = database

        self.create_index()

    def create_index(self):
        if self.collection_name in [c.name for c in self.client.get_collections().collections]:
            logger.info(f"Collection {self.collection_name} already exists.")
            return
        try:
            self.client.create_collection(
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

    def populate_index(self):
        batches: Iterator[list[dict]] = self._database.batch_generator(
            """SELECT d.id as doc_id, d.title_pt as title, 
                      d.abstract_pt as abstract, d.author, 
                      d.keywords_pt as keywords, e.vector, t.token, t.id as token_id, e.id as embeddings_id
               FROM embeddings AS e
               JOIN tokens AS t ON e.token_id = t.id
               JOIN documents AS d ON t.doc_id = d.id
               WHERE e.model_name = %s
                 AND t.token_type = %s
                 AND e.id NOT IN (SELECT embeddings_id FROM indexed);
            """,
            (self.model_name, self.token_type),
        )

        while True:
            for batch in batches:
                records: list[PointStruct] = []
                logger.info(f"Inserting records in batch of size: {len(batch)}")
                for record in batch:
                    records.append(
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
                    self.insert(records)
                    self._database.insert_many("""INSERT INTO indexed (embeddings_id) VALUES (%s)""",
                                                [(record['embeddings_id'],) for record in records])
                except Exception as e:
                    logger.error(f"NeuralIndex : {type(e)} : {e}")
                    self._database.insert_error(f"NeuralIndex : {type(e)} : {e}")

            time.sleep(3600)

    def insert(self, points: list[PointStruct]):
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        except (TimeoutError,
                qdrant_client.http.exceptions.ResponseHandlingException, httpcore.ReadTimeout, httpx.ReadTimeout):
            for point in points:
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=[point],
                    )
                except (TimeoutError,
                        qdrant_client.http.exceptions.ResponseHandlingException, httpcore.ReadTimeout,
                        httpx.ReadTimeout):
                    logger.error(f"Erro ao inserir ponto: {point}.")
                    self._database.insert_error(f"Erro ao inserir ponto: {point}.")


if __name__ == "__main__":
    database: Database = Database()
    indexer: NeuralIndex = NeuralIndex(database, list(config.MODELS.keys())[0], "sentence_with_keywords")
    indexer.populate_index()
