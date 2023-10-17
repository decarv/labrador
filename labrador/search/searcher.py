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
import asyncio
import heapq
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

import bs4
import httpx
import numpy as np
import qdrant_client
import requests
import sentence_transformers.util
import torch
from qdrant_client import models, grpc
from qdrant_client.http.models import ScoredPoint
from sentence_transformers import SentenceTransformer
from google.protobuf.json_format import MessageToDict

import config
from config import MODEL_CACHE_DIR
from ingest.models import Webpage
from util import utils, database, log

logger = log.configure_logger(__file__)
log = log.log(logger)


class SearchResultObject:
    def __init__(self, kwargs):
        self.doc_id = kwargs['doc_id']
        self.title = kwargs['title']
        self.abstract = kwargs['abstract']
        self.author = kwargs['author']
        self.url = kwargs['url']

    def to_dict(self):
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'abstract': self.abstract,
            'author': self.author,
            'url': self.url,
        }

    def __repr__(self):
        return f"<SearchResultObject {self.doc_id}>"


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

    def search(self, query: str, top_k: int = 10, _filters: dict = None) -> list[dict]:
        hits = self._retrieve(query, top_k)
        hits = self._filter(hits, _filters)
        hits = self._rank(hits)
        return hits


    @abstractmethod
    def _retrieve(self, query: str, top_k) -> list[dict]:
        raise NotImplementedError


    @abstractmethod
    def _rank(self, hits):
        raise NotImplementedError

    @abstractmethod
    def _filter(self, hits, _filters=None):
        raise NotImplementedError

    async def search_async(self, query: str, top_k: int = 10, _filters: dict = None) -> list[dict]:
        hits = await self._retrieve_async(query, top_k)
        hits = await self._filter_async(hits, _filters)
        hits = await self._rank_async(hits)
        return hits

    async def _retrieve_async(self, query: str, top_k) -> list[dict]:
        raise NotImplementedError

    async def _rank_async(self, hits):
        raise NotImplementedError

    async def _filter_async(self, hits, _filters=None):
        raise NotImplementedError


class RepositorySearcher(Searcher):
    def __init__(self):
        super(RepositorySearcher, self).__init__()

    def _retrieve(self, query: str, top_k) -> list[dict]:
        base_url: str = config.THESES_QUERY_URL.format(query.replace(" ", "%20"))
        documents_urls: list[str] = []
        page = 1

        start = time.time()
        while len(documents_urls) < top_k:
            url: str = base_url + str(page)
            response: requests.Response = requests.get(url)
            if response is not None:
                soup = bs4.BeautifulSoup(response.content, 'html.parser')
                divs = soup.find_all(
                    "div",
                    class_=["dadosLinha dadosCor1", "dadosLinha dadosCor2"]
                )
                documents_urls += [div.a['href'] for div in divs]
            page += 1
        end = time.time()
        logger.info("RS requests took: " + str(end - start) + " seconds")

        start = time.time()
        paths: list[str] = [Webpage.extract_path_suffix(url) for url in documents_urls]
        records: list[dict] = database.documents_query_url_path_suffix(paths)
        results: list[dict] = [record for record in records]
        end = time.time()
        logger.info("RS database took: " + str(end - start) + " seconds")

        return results

    async def _retrieve_async(self, query: str, top_k) -> list[dict]:
        base_url: str = config.THESES_QUERY_URL.format(query.replace(" ", "%20"))
        last_page = top_k // 10
        if top_k % 10 != 0:
            last_page += 1
        urls = [base_url + str(page) for page in range(1, last_page+1)]

        try:
            responses = await self.multi_fetch_async(urls)
            documents_paths: list[str] = []
            for i in range(last_page):
                response = responses[i]
                soup = bs4.BeautifulSoup(response.content, 'html.parser')
                divs = soup.find_all(
                    "div",
                    class_=["dadosLinha dadosCor1", "dadosLinha dadosCor2"]
                )
                documents_paths += [Webpage.extract_path_suffix(div.a['href']) for div in divs]
            records: list[dict] = await database.documents_query_url_path_suffix_async(documents_paths)
            results: list[dict] = [record for record in records]
        except httpx.ReadTimeout:
            logger.error("RS async took too long")
            results = []

        return results

    @staticmethod
    async def multi_fetch_async(urls: list[str]):
        async with httpx.AsyncClient(timeout=10.0) as client:
            responses = await asyncio.gather(*[client.get(url) for url in urls])
            return responses

    def _rank(self, hits):
        return hits

    def _filter(self, hits, _filters=None):
        return hits

    async def _rank_async(self, hits):
        return hits

    async def _filter_async(self, hits, _filters=None):
        return hits


class KeywordSearcher(Searcher):
    def __init__(self, *args, **kwargs):
        super(KeywordSearcher, self).__init__(*args, **kwargs)
        self.indexer = kwargs["indexer"]


class NeuralSearcher(Searcher):

    def __init__(self, client: qdrant_client.QdrantClient, *args, **kwargs):
        super(NeuralSearcher, self).__init__(*args, **kwargs)
        for attr in ["token_type"]:
            if not getattr(self, attr):
                raise ValueError(f"Missing required argument: {attr}.")

        self.client = client
        self.encoder_model = SentenceTransformer(self.model_name, device='cuda', cache_folder=MODEL_CACHE_DIR)
        self.collection_name = utils.collection_name(self.model_name, self.token_type)

    def _retrieve(self, query: str, top_k: int) -> list[dict]:
        vector: np.ndarray = self.encoder_model.encode(query, show_progress_bar=False, convert_to_tensor=True)
        hits: list[ScoredPoint] = self.client.search(
            collection_name=self.collection_name,
            search_params=models.SearchParams(
                hnsw_ef=128,
                exact=False
            ),
            query_vector=vector,
            limit=10,
            with_payload=True
        )
        results = [hit.model_dump()['payload'] for hit in hits]
        return results

    def _filter(self, hits, _filters=None):
        return hits

    def _rank(self, hits):
        return hits

    async def _retrieve_async(self, query: str, top_k) -> list[dict[str, Any]]:
        """

        limit:
        There are around 10 vectors for each document in the collection that point to a sentence
        in the document. We want to retrieve the top_k documents, so we need to retrieve top_k * 10
        vectors on average, to make sure we have enough vectors to retrieve the top_k documents.
        """

        limit: int = top_k * 10
        vector: np.ndarray = self.encoder_model.encode(query, show_progress_bar=False, convert_to_tensor=True)
        response_msg = await self.client.async_grpc_points.Search(
            grpc.SearchPoints(
                collection_name=self.collection_name,
                vector=vector,
                limit=limit,
                with_payload=grpc.WithPayloadSelector(enable=True),
                with_vectors=grpc.WithVectorsSelector(enable=False),
            )
        )
        response_dict = MessageToDict(response_msg)
        response_result = response_dict['result']
        hits = []
        inserted_hits = set()
        for hit in response_result:
            hit_dict = self.protobuf_to_dict(hit)
            if hit_dict['doc_id'] not in inserted_hits:
                inserted_hits.add(hit_dict['doc_id'])
                hits.append(hit_dict)
            if len(hits) == top_k:
                break
        return hits

    @staticmethod
    def protobuf_to_dict(protobuf):
        dictionary = {}
        for key, value_dict in protobuf['payload'].items():
            value_fmt = list(value_dict.keys())[0]
            if value_fmt == 'stringValue':
                dictionary[key] = value_dict[value_fmt]
            elif value_fmt == 'integerValue':
                dictionary[key] = int(value_dict[value_fmt])
            else:
                raise ValueError(f"Unknown value format: {value_fmt}")
        return dictionary

    async def _rank_async(self, hits):
        return hits

    async def _filter_async(self, hits, _filters=None):
        return hits


class LocalNeuralSearcher(Searcher):
    """
    The LocalNeuralSearcher is a searcher that uses a local infrastructure to retrieve and rank results.
    """
    def __init__(self, *args, **kwargs):
        super(LocalNeuralSearcher, self).__init__(*args, **kwargs)

        for attr in ["language", "token_type"]:
            if not getattr(self, attr):
                raise ValueError(f"Missing required argument: {attr}.")

        self.encoder_model = SentenceTransformer(self.model_name, device='cuda', cache_folder=config.MODEL_CACHE_DIR)

    def _retrieve(self, query: str, top_k, similarity_func=sentence_transformers.util.dot_score) -> dict:
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


query = "ataque cardíaco criança"


async def main_async():
    await database.async_conn_pool_init()

    rs = RepositorySearcher()
    start = time.time()
    hit_list = await rs.search_async(query, 10)
    end = time.time()
    print("RS Async: ", end - start)

    ns = NeuralSearcher(
        client=qdrant_client.QdrantClient(url=config.QDRANT_HOST, port=config.QDRANT_PORT),
        model_name=list(config.MODELS.keys())[0],
        token_type="sentence_with_keywords",
        language="pt",
    )
    start = time.time()
    hit_list = await ns.search_async(query, 10)
    end = time.time()
    print("NS Async: ", end - start)


def main():
    database.conn_pool_init()

    rs = RepositorySearcher()
    start = time.time()
    hit_list = rs.search(query)
    end = time.time()
    print("RS: ", end - start)

    ns = NeuralSearcher(
        client=qdrant_client.QdrantClient(url=config.QDRANT_HOST, port=config.QDRANT_PORT),
        model_name=list(config.MODELS.keys())[0],
        token_type="sentence_with_keywords",
        language="pt",
    )
    start = time.time()
    hit_list = ns.search(query)
    end = time.time()
    print("NS: ", end - start)


if __name__ == "__main__":
    main()
    asyncio.run(main_async())
