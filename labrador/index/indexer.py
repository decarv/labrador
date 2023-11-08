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
from typing import Iterator

import httpcore
import httpx
import psycopg2
import qdrant_client.http.exceptions
from httpcore import ReadTimeout
from psycopg2.extras import RealDictCursor, RealDictRow
import weaviate
import weaviate.gql.get

import config
from util import log, database
from util.utils import collection_name

from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

BATCH_SIZE: int = 16

logger = log.configure_logger(__file__)
log = log.log(logger)


class SparseIndexer:
    pass


class QdrantIndexer:
    def __init__(self,
                 model_name: str, token_type: str,
                 vectorizer: str = 'none'):
        self.client = QdrantClient(config.QDRANT_HOST, port=config.QDRANT_PORT)
        self.collection_name = collection_name(model_name, token_type)
        self.model_name = model_name
        self.token_type = token_type

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
                    database.errors_insert(f"Erro ao inserir ponto: {point}.")
        except Exception as e:
            logger.error(f"Erro ao inserir batch: {e}.")

    def insert_records(self, records: list[RealDictRow]):
        points: list[PointStruct] = [self.create_object(record) for record in records]
        self.insert(points)

    def populate_index(self):
        with psycopg2.connect(
                dbname=config.POSTGRESQL_DB_NAME,
                user=config.POSTGRESQL_DB_USER,
                password=config.POSTGRESQL_DB_PASSWORD,
                host=config.POSTGRESQL_DB_HOST,
                port=config.POSTGRESQL_DB_PORT,
        ) as conn:
            for i, records in enumerate(self.records_generator(conn)):
                logger.info(f"Iniciando batch...{i}")
                points: list[PointStruct] = [self.create_object(record) for record in records]
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                    )
                    self._update_embeddings_in_qdrant(conn, records)
                except (TimeoutError,
                        qdrant_client.http.exceptions.ResponseHandlingException, httpcore.ReadTimeout, httpx.ReadTimeout):
                    for i, point in enumerate(points):
                        try:
                            self.client.upsert(
                                collection_name=self.collection_name,
                                points=[point],
                            )
                            self._update_embeddings_in_qdrant(conn, [records[i]])
                        except (TimeoutError,
                                qdrant_client.http.exceptions.ResponseHandlingException, httpcore.ReadTimeout,
                                httpx.ReadTimeout):
                            logger.error(f"Erro ao inserir ponto: {point}.")
                            database.errors_insert(f"Erro ao inserir ponto: {point}.")

                except Exception as e:
                    logger.error(f"Erro ao inserir batch: {e}.")

    @staticmethod
    def create_object(record: dict) -> PointStruct:
        data_object = PointStruct(
            id=record['embedding_id'],
            vector=record['embeddings'],
            payload={
                "doc_id": record['doc_id'],
                "title": record['title'],
                "abstract": record['abstract'],
                "keywords": record['keywords'],
                "author": record['author'],
                "token": record['token'],
            }
        )
        return data_object

    def records_generator(self, conn):
        select_query: str = """
                        SELECT d.id as doc_id, d.title_pt as title, 
                               d.abstract_pt as abstract, d.author, 
                               d.keywords_pt as keywords, e.embeddings, t.token, e.id as embedding_id
                        FROM embeddings AS e
                        JOIN tokens AS t ON e.token_id = t.id
                        JOIN clean_data AS d ON t.data_id = d.id
                        WHERE e.model_name = %s
                          AND t.token_type = %s
                          AND e.in_qdrant = False
                        LIMIT %s
                        """

        model_name = self.model_name
        token_type = self.token_type
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            while True:
                cur.execute(select_query, (model_name, token_type, BATCH_SIZE))
                records = cur.fetchall()
                if not records:
                    break
                yield records

    @log
    def _update_embeddings_in_qdrant(self, conn, records: list[RealDictRow]):
        update_query: str = """
                        UPDATE embeddings
                        SET in_qdrant = True
                        WHERE id = %s
                        """
        update_vars_list = [(record['embedding_id'],) for record in records]
        with conn.cursor() as cur:
            cur.executemany(update_query, update_vars_list)
            conn.commit()

    @log
    def _get_added_ids(self) -> set[int]:
        query: weaviate.gql.get.GetBuilder = (
            self.client.query.get(self.class_name, ["doc_id"])
            .with_additional(["id"])
            .with_limit(1024)
        )
        added_ids: set[int] = set()
        query_result: dict = query.do()
        while query_result:
            query_result: dict = query.do()
            records: list[dict] = query_result['data']['Get'][self.class_name.capitalize()]
            if not records:
                break
            for record in records:
                added_ids.add(record['doc_id'])
            last_uuid = records[-1]['_additional']['id']
            query = (
                self.client.query.get(self.class_name, ["doc_id"])
                .with_additional(["id"])
                .with_after(last_uuid)
                .with_limit(1024)
            )
        return added_ids
#
#
# class WeaviateIndexer:
#     def __init__(self, client: weaviate.Client,
#                  model_name: str, token_type: str,
#                  description: str = "", vectorizer: str = 'none'):
#         self.client = client
#         self.class_name = collection_name(model_name, token_type)
#         logger.debug(f"Creating class {self.class_name}...")
#         self.model_name = model_name
#         self.token_type = token_type
#         self.class_object: dict = {
#             "class": self.class_name,
#             "description": description,
#             "vectorizer": vectorizer,
#             "properties": [
#                 {
#                     "name": "doc_id",
#                     "dataType": ["int"],
#                 },
#                 {
#                     "name": "embedding_id",
#                     "dataType": ["int"],
#                 },
#                 {
#                     "name": "title",
#                     "dataType": ["string"],
#                 },
#                 {
#                     "name": "abstract",
#                     "dataType": ["string"],
#                 },
#                 {
#                     "name": "keywords",
#                     "dataType": ["string"],
#                 },
#                 {
#                     "name": "author",
#                     "dataType": ["string"],
#                 },
#                 {
#                     "name": "token",
#                     "dataType": ["string"],
#                 },
#             ],
#             "vectorIndexType": "hnsw",
#             "vectorIndexConfig": {
#                 "distance": config.MODELS[model_name],
#             }
#         }
#         try:
#             self.client.schema.create_class(self.class_object)
#         except Exception as e:
#             logger.warning(e)
#
#     @log
#     def create_index(self):
#         with self.client.batch(batch_size=BATCH_SIZE, num_workers=1) as batch:
#             for i, records in enumerate(self.records_generator()):
#                 logger.info(f"Iniciando batch...{i}")
#                 for record in records:
#                     logger.info(f"Adicionando objeto de id {record['embedding_id']} ao batch...")
#                     batch.add_data_object(
#                         record,
#                         class_name=self.class_name,
#                         vector=record['embeddings'],
#                     )
#
#     @staticmethod
#     def _create_object(record: RealDictRow) -> dict:
#         data_object = {
#             "doc_id": record['doc_id'],
#             "embedding_id": record['embedding_id'],
#             "title": record['title'],
#             "abstract": record['abstract'],
#             "keywords": record['keywords'],
#             "author": record['author'],
#             "token": record['token'],
#         }
#         return data_object
#
#     def records_generator(self):
#         select_query: str = """
#                         SELECT d.id as doc_id, d.title_pt as title,
#                                d.abstract_pt as abstract, d.author,
#                                d.keywords_pt as keywords, e.embeddings, t.token, e.id as embedding_id
#                         FROM embeddings AS e
#                         JOIN tokens AS t ON e.token_id = t.id
#                         JOIN data AS d ON t.data_id = d.id
#                         WHERE e.model_name = %s
#                           AND t.token_type = %s
#                           AND e.in_weaviate = False
#                         LIMIT %s
#                         """
#
#         model_name = self.model_name
#         token_type = self.token_type
#         with psycopg2.connect(
#             dbname=config.POSTGRESQL_DB_NAME,
#             user=config.POSTGRESQL_DB_USER,
#             password=config.POSTGRESQL_DB_PASSWORD,
#             host=config.POSTGRESQL_DB_HOST,
#             port=config.POSTGRESQL_DB_PORT,
#         ) as conn:
#             with conn.cursor(cursor_factory=RealDictCursor) as cur:
#                 while True:
#                     cur.execute(
#                         select_query,
#                         (model_name, token_type, BATCH_SIZE)
#                     )
#                     records = cur.fetchall()
#                     if not records:
#                         break
#                     yield records
#
#                     self._update_embeddings_in_weaviate(conn, records)
#
#     @log
#     def _update_embeddings_in_weaviate(self, conn, records: list[RealDictRow]):
#         update_query: str = """
#                         UPDATE embeddings
#                         SET in_weaviate = True
#                         WHERE id = %s
#                         """
#         update_vars_list = [(record['embedding_id'],) for record in records]
#         logger.info(f"Updating {len(update_vars_list)} embeddings in Weaviate...")
#         with conn.cursor() as cur:
#             cur.executemany(update_query, update_vars_list)
#             conn.commit()
#
#     @log
#     def _get_added_ids(self) -> set[int]:
#         query: weaviate.gql.get.GetBuilder = (
#             self.client.query.get(self.class_name, ["doc_id"])
#             .with_additional(["id"])
#             .with_limit(1024)
#         )
#         added_ids: set[int] = set()
#         query_result: dict = query.do()
#         while query_result:
#             query_result: dict = query.do()
#             records: list[dict] = query_result['data']['Get'][self.class_name.capitalize()]
#             if not records:
#                 break
#             for record in records:
#                 added_ids.add(record['doc_id'])
#             last_uuid = records[-1]['_additional']['id']
#             query = (
#                 self.client.query.get(self.class_name, ["doc_id"])
#                 .with_additional(["id"])
#                 .with_after(last_uuid)
#                 .with_limit(1024)
#             )
#         return added_ids


if __name__ == '__main__':
    print("The usage of indexer.py as a script is deprecated.")
    # model_name = list(config.MODELS.keys())[0]
    # token_type = "sentence_with_keywords"
    # indexer = QdrantIndexer(
    #     model_name=model_name,
    #     token_type=token_type,
    # )
    # # indexer.create_index()
    # indexer.populate_index()
    #
# from annoy import AnnoyIndex
# search_index = AnnoyIndex(embeds.shape[1], 'angular')
# # Add all the vectors to the search index
# for i in range(len(embeds)):
#     search_index.add_item(i, embeds[i])
#
# search_index.build(10) # 10 trees
# search_index.save('test.ann')
# pd.set_option('display.max_colwidth', None)
#
#
# def search(query):
#     # Get the query's embedding
#     query_embed = co.embed(texts=[query]).embeddings
#
#     # Retrieve the nearest neighbors
#     similar_item_ids = search_index.get_nns_by_vector(query_embed[0],
#                                                       3,
#                                                       include_distances=True)
#     # Format the results
#     results = pd.DataFrame(data={'texts': texts[similar_item_ids[0]],
#                                  'distance': similar_item_ids[1]})
#
#     print(texts[similar_item_ids[0]])
#
#     return results
