import heapq
import json
import os
from typing import Any
import gc
import asyncio

import numpy as np
import qdrant_client
import sentence_transformers.util
import torch
from google.protobuf.json_format import MessageToDict
from qdrant_client import models, grpc
from qdrant_client.http.models import ScoredPoint
from sentence_transformers import SentenceTransformer
from torch.cuda import OutOfMemoryError

from labrador import config
from labrador.config import MODEL_CACHE_DIR
from labrador.models import Searcher
from labrador.util import utils, database, log
from labrador.util.log import logger


class DenseSearcher(Searcher):

    def __init__(self, client: qdrant_client.QdrantClient, *args, **kwargs):
        super(DenseSearcher, self).__init__(*args, **kwargs)
        for attr in ["token_type"]:
            if not getattr(self, attr):
                raise ValueError(f"Missing required argument: {attr}.")

        self.client = client
        self.encoder_model = SentenceTransformer(self.model_name, device='cuda', cache_folder=MODEL_CACHE_DIR)
        self.collection_name = utils.collection_name(self.model_name, self.token_type)

    def _retrieve(self, query: str, top_k: int) -> list[dict]:
        limit: int = top_k * 10
        try:
            vector: np.ndarray = self.encoder_model.encode(query, show_progress_bar=False, convert_to_tensor=True)
        except OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            vector: np.ndarray = self.encoder_model.encode(query, show_progress_bar=False, convert_to_tensor=True)

        response_msg = self.client.grpc_points.Search(
            grpc.SearchPoints(
                collection_name=self.collection_name,
                vector=vector,
                limit=limit,
                with_payload=grpc.WithPayloadSelector(enable=True),
                with_vectors=grpc.WithVectorsSelector(enable=False),
            )
        )
        return self._extract_hits_from_message(response_msg, top_k)


        # limit: int = top_k * 10  # TODO: is there a better fix?

#         vector: np.ndarray = self.encoder_model.encode(query, show_progress_bar=False, convert_to_tensor=True)
#         response_msg = self.client.grpc_points.Search(
#             grpc.SearchPoints(
#                 collection_name=self.collection_name,
#                 vector=vector,
#                 limit=limit,
# #                 with_payload=grpc.WithPayloadSelector(enable=True),
# #                 with_vectors=grpc.WithVectorsSelector(enable=False),
#             )
#         )
# #         response_dict = MessageToDict(response_msg)
# #         response_result = response_dict['result']
#         hits = []
#         inserted_hits = set()
#         unique_property = 'doc_id'
#         for hit in response_result:
#             hit_dict = self.protobuf_to_dict(hit)
#             if hit_dict[unique_property] not in inserted_hits:
#                 inserted_hits.add(hit_dict[unique_property])
#                 hits.append(hit_dict)
#             if len(hits) == top_k:
#                 break
#         return hits

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
        try:
            vector: np.ndarray = self.encoder_model.encode(query, show_progress_bar=False, convert_to_tensor=True)
        except OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
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
        return self._extract_hits_from_message(response_msg, top_k)

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

    def _extract_hits_from_message(self, response_msg, top_k):
        response_dict = MessageToDict(response_msg)
        if 'result' not in response_dict:
                    return []
        response_result = response_dict['result']
        hits = []
        inserted_hits = set()
        unique_property = 'doc_id'
        for hit in response_result:
            hit_dict = self.protobuf_to_dict(hit)
            if hit_dict[unique_property] not in inserted_hits:
                inserted_hits.add(hit_dict[unique_property])
                hits.append(hit_dict)
                if len(hits) == top_k:
                    break

        return hits

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
                config.REPORTS_DIR,
                f"{__class__.__name__}_{normalized_model_name}_{self.token_type}_{self.language}.json"), "a") as f:
            f.write(json.dumps(hits))


if __name__ == "__main__":
    from config import QDRANT_HOST, QDRANT_GRPC_PORT
    import asyncio
    async def main():
        query = "O que é a covid-19?"
        top_k = 10

        client = qdrant_client.QdrantClient(QDRANT_HOST, port=QDRANT_GRPC_PORT)
        ns = DenseSearcher(
            client=client,
            model_name=list(config.MODELS.keys())[0],
            token_type="sentence_with_keywords",
            language="pt",
        )
        print(await ns.search_async(query, top_k))

    asyncio.run(main())
