"""
searcher

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

import os
import heapq
import logging
from typing import Union

import numpy as np
import sentence_transformers.util
import torch
from psycopg2.extras import RealDictRow
from sentence_transformers import SentenceTransformer
import json
import weaviate
from abc import ABC, abstractmethod
from sentence_transformers.util import semantic_search


import config
import util.database
from util import utils, database, log

logger = log.configure_logger(__file__)
log = log.log(logger)


class Searcher(ABC):
    def __init__(self, *args, **kwargs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            logger.warning("CUDA not available. Using CPU instead.")

        self.model_name = kwargs.get('model_name')
        self.collection_name = kwargs.get('collection_name')
        self.language = kwargs.get('language')
        self.token_type = kwargs.get('token_type')
        if not self.token_type:
            self.token_type = kwargs.get('units_type')

    def search(self, query: str, _filters: dict = None) -> list[dict]:
        hits = self._retrieve(query)
        hits = self._filter(hits, _filters)
        hits = self._rank(hits)
        return hits

    @abstractmethod
    def _retrieve(self, query: str, top_k: int = 30) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def _rank(self, hits):
        raise NotImplementedError

    @abstractmethod
    def _filter(self, hits, _filters=None):
        raise NotImplementedError

    @abstractmethod
    def _hits_save(self, query: str, hits: str):
        pass


class KeywordSearcher(Searcher):
    def __init__(self, *args, **kwargs):
        super(KeywordSearcher, self).__init__(*args, **kwargs)
        self.indexer = kwargs["indexer"]


class NeuralSearcher(Searcher):
    pass


class LocalNeuralSearcher(Searcher):
    """
    The LocalNeuralSearcher is a searcher that uses a local infrastructure to retrieve and rank results.
    """
    def __init__(self, *args, **kwargs):
        super(LocalNeuralSearcher, self).__init__(*args, **kwargs)

        for attr in ["language", "token_type"]:
            if not getattr(self, attr):
                raise ValueError(f"Missing required argument: {attr}.")

        self.encoder_model = SentenceTransformer(
            self.model_name, device='cuda', cache_folder=config.MODEL_CACHE_DIR
        )

    def _retrieve(self, query: str, top_k: int = 30, similarity_func=sentence_transformers.util.dot_score) -> dict:
        query_embeddings: torch.tensor = self.embed_query(query)
        corpus_embeddings_generator: torch.tensor = database.embeddings_read(
            self.model_name, self.token_type, self.language, batch_size=128
        )
        hits: list[list[float, int]] = []  # min heap
        hits_dict: dict[int, list[int, float]] = {}
        for batch in corpus_embeddings_generator:
            logger.info("Processing batch...")
            embeddings_list = [torch.tensor(r['embeddings']) for r in batch]
            corpus_embeddings: torch.tensor = torch.stack(embeddings_list)
            qe = query_embeddings.to(self.device)
            ce = corpus_embeddings.to(self.device)
            scores: list[float] = similarity_func(qe, ce)[0]
            for i, score in enumerate(scores):
                data_id = batch[i]['data_id']
                if data_id in hits_dict and score > hits_dict[data_id][0]:
                    new_hits = []
                    new_hits_dict = {}
                    while len(hits) > 0:
                        hit: list[float, int] = heapq.heappop(hits)
                        if hit[1] != data_id:
                            heapq.heappush(new_hits, hit)
                            new_hits_dict[hit[1]] = hit
                    hits = new_hits
                    hits_dict = new_hits_dict
                elif data_id in hits_dict and score < hits_dict[data_id][0]:
                    continue

                if len(hits) < top_k or score > hits[0][0]:
                    hit = [score, data_id]
                    if len(hits) == top_k:
                        rm_id = hits[0][1]
                        heapq.heapreplace(hits, hit)
                        del hits_dict[rm_id]
                    else:
                        heapq.heappush(hits, hit)
                    hits_dict[data_id] = hit
                assert len(hits_dict) == len(hits)

        hits = [heapq.heappop(hits) for _ in range(len(hits))][::-1]
        data_ids = []
        hits_data: dict = {}
        for hit in hits:
            score, data_id = hit
            data_ids.append(data_id)
            hits_data[data_id] = {'score': score}

        data = database.data_read_in_ids(data_ids)
        for record in data:
            hits_data[record['id']]['data'] = record

        return hits_data

    def process_query(self, query):
        return query

    def embed_query(self, query):
        vector: np.ndarray = self.encoder_model.encode(query, show_progress_bar=False, convert_to_tensor=True)
        return vector

    def _rank(self, hits):
        return hits

    def _filter(self, hits, _filters=None):
        return hits

    def _hits_save(self, query: str, hits: str):
        pass

    def save(self, hits: list[dict[str]]):
        normalized_model_name = self.model_name.replace("/", "-")
        with open(os.path.join(
                config.RESULTS_DIR,
                f"{__class__.__name__}_{normalized_model_name}_{self.token_type}_{self.language}.json"), "a") as f:
            f.write(json.dumps(hits))

#
# class KeywordSearcher(Searcher):
#     def __init__(self, *args, **kwargs):
#         super(KeywordSearcher, self).__init__(*args, **kwargs)
#         auth_config = weaviate.auth.AuthApiKey(
#             api_key=os.environ['WEAVIATE_API_KEY'])
#
#         client = weaviate.Client(
#             url=os.environ['WEAVIATE_API_URL'],
#             auth_client_secret=auth_config,
#             additional_headers={
#                 "X-Cohere-Api-Key": os.environ['COHERE_API_KEY'],
#             }
#         )
#
#
#     def _keyword_search(self, query,
#                        results_lang='en',
#                        properties=["title", "url", "text"],
#                        num_results=3):
#         where_filter = {
#             "path": ["lang"],
#             "operator": "Equal",
#             "valueString": results_lang
#         }
#
#         response = (
#             client.query.get("Articles", properties)
#             .with_bm25(
#                 query=query
#             )
#             .with_where(where_filter)
#             .with_limit(num_results)
#             .do()
#         )
#
#         result = response['data']['Get']['Articles']
#         return result
#
#
# class DenseSearcher(Searcher):
#     """
#     TODO: documentation
#     Retrieve + Rank
#     """
#     def __init__(self, *args, **kwargs):
#         super(DenseSearcher, self).__init__(args, kwargs)
#         self.client = kwargs['qdrant_client']
#
#     @log
#     def _retrieve(self, query: str, hits_cnt=30) -> List[dict]:
#         """
#         This function is responsible for processing the query, retrieving and ranking
#         the results of the retrieval.
#         :param query:
#         :param filter_:
#         :return:
#         """
#         vector = self.process_query(query)
#         hits = self.client.search(
#             collection_name=self.collection_name,
#             query_vector=vector,
#             query_filter=Filter(**filter_) if filter_ else None,
#             top=30
#         )
#         hits = self._rank(hits)
#         return [hit.payload for hit in hits]
#
#     @log
#     def process_query(self, query):
#         """
#         For now, this returns the direct encoding of the model. This pre-processing step
#         for queries may consider standardization, tokenization and spell check.
#         :param query:
#         :return:
#         """
#         vector: list = self.encoder_model.encode(query).tolist()
#         return vector
#
#     @log
#     def _rank(self, hits):
#         """
#         For now, this returns the same order as the hits.
#         :param hits: List of computed nearest vectors.
#         :return:
#         """
#         return hits
#
#     @log
#     def _filter(self, hits, _filters=None):
#         """
#         Responsible for filtering invalid hits amongst all hits found with ANN or knowledge graph.
#         :param hits:
#         :return:
#         """
#         pass
#


if __name__ == "__main__":
    model = config.MODELS[0]
    query = "ataque cardíaco"
    database.initialize_conn_pool(4, 10)
    ls = LocalNeuralSearcher(
        model_name=model,
        token_type="sentence_with_keywords",
        language="pt",
    )
    hit_list = ls.search(query)
    print(hit_list)
