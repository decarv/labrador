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
from abc import ABC, abstractmethod
import torch
from util import log

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
    class SearcherError(Exception):
        pass

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
