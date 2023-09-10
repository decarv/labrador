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
import logging
import numpy as np
import torch
from psycopg2.extras import RealDictRow
from sentence_transformers import SentenceTransformer
from utils import log
import utils
import config
from sentence_transformers.util import cos_sim
from abc import ABC, abstractmethod

logger = utils.configure_logger(__name__)


class Searcher(ABC):
    def __init__(self, *args, **kwargs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            logging.warning("CUDA not available. Using CPU instead.")

        self.model_name = kwargs.get('model_name')
        self.collection_name = kwargs.get('collection_name')
        self.language = kwargs.get('language')
        self.units_type = kwargs.get('units_type')

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


class LocalSearcher(Searcher):
    def __init__(self, *args, **kwargs):
        super(LocalSearcher, self).__init__(*args, **kwargs)
        self.encoder_model = SentenceTransformer(
            self.model_name, device='cuda', cache_folder=config.MODEL_CACHE_DIR, convert_to_tensor=True
        )
        self.embeddings: torch.tensor = utils.embeddings_load(self.model_name, self.units_type, self.language)
        self.data: list[RealDictRow] = utils.metadata_read()
        self.imap: list[int] = utils.indices_load(self.units_type, self.language)

    @log(logger)
    def _retrieve(self, query: str, top_k: int = 30) -> list[dict]:
        tensor: torch.tensor = self.process_query(query)
        [scores] = cos_sim(tensor, self.embeddings)
        hits = np.argsort(scores)[::-1][:top_k]
        return [self.data[self.imap[i]] for i in hits]

    @log(logger)
    def process_query(self, query):
        vector: np.ndarray = self.encoder_model.encode(query, show_progress_bar=False)
        return vector

    @log(logger)
    def _rank(self, hits):
        return hits

    @log(logger)
    def _filter(self, hits, _filters=None):
        return hits

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
