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

import psycopg2
from psycopg2.extras import RealDictCursor, RealDictRow
import weaviate
import weaviate.gql.get

from index.tokenizer import Tokenizer
import config
from util import log
from util.utils import weaviate_class_name

BATCH_SIZE: int = 64

logger = log.configure_logger(__file__)
log = log.log(logger)


class SparseIndexer:
    pass


class DenseIndexer:
    def __init__(self, client: weaviate.Client,
                 model_name: str, token_type: str,
                 description: str = "", vectorizer: str = 'none'):
        self.client = client
        self.class_name = weaviate_class_name(model_name, token_type)
        logger.debug(f"Creating class {self.class_name}...")
        self.model_name = model_name
        self.token_type = token_type
        self.class_object: dict = {
            "class": self.class_name,
            "description": description,
            "vectorizer": vectorizer,
            "properties": [
                {
                    "name": "doc_id",
                    "dataType": ["int"],
                },
                {
                    "name": "embedding_id",
                    "dataType": ["int"],
                },
                {
                    "name": "title",
                    "dataType": ["string"],
                },
                {
                    "name": "abstract",
                    "dataType": ["string"],
                },
                {
                    "name": "keywords",
                    "dataType": ["string"],
                },
                {
                    "name": "author",
                    "dataType": ["string"],
                },
                {
                    "name": "token",
                    "dataType": ["string"],
                },
            ],
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": {
                "distance": config.MODELS[model_name],
            }
        }
        try:
            self.client.schema.create_class(self.class_object)
        except Exception as e:
            logger.warning(e)

    @log
    def create_index(self):
        with self.client.batch(batch_size=BATCH_SIZE, num_workers=4) as batch:
            for i, records in enumerate(self.records_generator()):
                logger.info(f"Iniciando batch...{i}")
                for record in records:
                    batch.add_data_object(
                        record,
                        class_name=self.class_name,
                        vector=record['embeddings'],
                    )

    @staticmethod
    def _create_object(record: RealDictRow) -> dict:
        data_object = {
            "doc_id": record['doc_id'],
            "embedding_id": record['embedding_id'],
            "title": record['title'],
            "abstract": record['abstract'],
            "keywords": record['keywords'],
            "author": record['author'],
            "token": record['token'],
        }
        return data_object

    def records_generator(self):
        select_query: str = """
                        SELECT d.id as doc_id, d.title_pt as title, 
                               d.abstract_pt as abstract, d.author, 
                               d.keywords_pt as keywords, e.embeddings, t.token, e.id as embedding_id
                        FROM embeddings AS e
                        JOIN tokens AS t ON e.token_id = t.id
                        JOIN data AS d ON t.data_id = d.id
                        WHERE e.model_name = %s
                          AND t.token_type = %s
                        LIMIT %s
                        """

        model_name = self.model_name
        token_type = self.token_type
        with psycopg2.connect(
            dbname=config.POSTGRESQL_DB_NAME,
            user=config.POSTGRESQL_DB_USER,
            password=config.POSTGRESQL_DB_PASSWORD,
            host=config.POSTGRESQL_DB_HOST,
            port=config.POSTGRESQL_DB_PORT,
        ) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                while True:
                    cur.execute(
                        select_query,
                        (model_name, token_type, BATCH_SIZE)
                    )
                    records = cur.fetchall()
                    if not records:
                        break
                    yield records

                    self._update_embeddings_in_weaviate(conn, records)

    def _update_embeddings_in_weaviate(self, conn, records: list[RealDictRow]):
        update_query: str = """
                        UPDATE embeddings
                        SET in_weaviate = True
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


if __name__ == '__main__':
    for model_name in config.MODELS:
        for token_type in Tokenizer.token_types()[::-1]:
            DenseIndexer(
                client=weaviate.Client(config.WEAVIATE_URL),
                model_name=model_name,
                token_type=token_type,
            ).create_index()

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
