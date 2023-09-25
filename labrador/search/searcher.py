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
from typing import Union

import numpy as np
import torch
from psycopg2.extras import RealDictRow
from sentence_transformers import SentenceTransformer
import json
from abc import ABC, abstractmethod
from sentence_transformers.util import semantic_search

import config
import util.database
import util.log
from util import utils
from util import log

logger = util.log.configure_logger(__file__)
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


class LocalNeuralSearcher(Searcher):
    """
    NeuralSearcher implemented in version 0.1.0.
    The LocalNeuralSearcher is a searcher that uses a local infrastructure to retrieve and rank results.
    The indices and embeddings are loaded from disk into memory.

    The function used to train the model is the Encoder's current local_encode() function.
    """
    def __init__(self, *args, **kwargs):
        super(LocalNeuralSearcher, self).__init__(*args, **kwargs)

        for attr in ["language", "token_type"]:
            if not getattr(self, attr):
                raise ValueError(f"Missing required argument: {attr}.")

        self.encoder_model = SentenceTransformer(
            self.model_name, device='cuda', cache_folder=config.MODEL_CACHE_DIR
        )
        self.data: list[RealDictRow] = kwargs.get('data')
        self.tokenized_data: list[RealDictRow] = kwargs.get('training_data')
        self.indices, self.token_indices = utils.indices_load(self.token_type, self.language)
        self.corpus_embeddings: torch.tensor = utils.embeddings_load(self.model_name, self.token_type, self.language)

    def _get_token_indices(self) -> list[int]:
        """
        Example:
            indices: [5 5 5 6 6 7 7 7 7 8 9 9 9]
            token indices: [0 1 2 0 1 0 1 2 3 0 0 1 2]
        """
        token_indices: list[int] = []
        curr_idx: int = -1
        token_idx: int = 0
        for idx in self.indices:
            if idx != curr_idx:
                token_idx = 0
                curr_idx = idx
            token_indices.append(token_idx)
            token_idx += 1
        return token_indices

    @log(logger)
    def _retrieve(self, query: str, top_k: int = 30) -> list[dict]:
        """

        self.indices is a map from the index of the embedding to the index of the data.

        :param query:
        :param top_k:
        :return:
        """
        query: str = self.process_query(query)
        query_embeddings: torch.tensor = self.embed_query(query)
        queries_results: list[list[dict[str, Union[int, float]]]] = semantic_search(
            query_embeddings=query_embeddings,
            corpus_embeddings=self.corpus_embeddings,
            top_k=top_k
        )
        if not self.data and not self.tokenized_data:
            raise ValueError("No data or tokenized data provided for searcher.")
        hits: list[dict] = []
        for results in queries_results:
            for result in results:
                corpus_id = result['corpus_id']
                score = result['score']
                data_index = self.indices[corpus_id]
                title = self.data[data_index][f'title_{self.language}']
                abstract = self.data[data_index][f'abstract_{self.language}']
                keywords = self.data[data_index][f'keywords_{self.language}']

                token_index = self.token_indices[data_index]
                token_hit = self.tokenized_data[data_index][f"title_tokens_{self.language}"][token_index]
                hits.append(
                    {
                        'score': score,
                        'data': {
                            'title': title,
                            'abstract': abstract,
                            'keywords': keywords
                        },
                        'token_hit': token_hit
                    }
                )
        return hits

    def process_query(self, query):
        return query

    @log(logger)
    def embed_query(self, query):
        vector: np.ndarray = self.encoder_model.encode(query, show_progress_bar=False, convert_to_tensor=True)
        return vector

    @log(logger)
    def _rank(self, hits):
        return hits

    @log(logger)
    def _filter(self, hits, _filters=None):
        return hits

    def _hits_save(self, query: str, hits: str):
        pass

    def save(self, hits: list[dict[str]]):
        with open(os.path.join(
                config.RESULTS_DIR,
                f"search_results_{self.model_name}_{self.token_type}_{self.language}.json"
        ), "w") as f:
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
    query = "computação em nuvem"
    for model in config.MODELS:
        ls = LocalNeuralSearcher(
            model_name=model,
            token_type="sentence_with_keywords",
            language="pt",
            data=util.database.table_read("clean_metadata"),
            training_data=util.database.table_read("clean_tokenized_metadata")
        )
        _hits = ls.search(query)
        ls.save(_hits)
