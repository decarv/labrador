"""
tokenizer

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
import gc
import os
import pickle
from typing import Optional, Callable, Iterator
from psycopg2.extras import RealDictRow, RealDictCursor

import utils
import config

logger = utils.configure_logger(__name__)


class Tokenizer:
    def __init__(self, token_type: str, language: str = "pt"):
        _token_generator_name: str = f"_generate_{token_type}_tokens"
        if not hasattr(self, _token_generator_name):
            raise ValueError(f"Invalid token type: {token_type}. Valid token types are: {Tokenizer.token_types()}")

        self._tokenizer: Callable = getattr(self, _token_generator_name)
        self.token_type = token_type
        self.language = language
        self.path = utils.tokens_path(self.token_type, self.language)
        self.indices_map_path = utils.indices_path(self.token_type, self.language)

        self.tokens: Optional[list[list[str]]] = None
        self.indices_map: Optional[list[int]] = None
        self.token_indices_map: Optional[list[int]] = None

    def tokenize_async(self, data_chunks: Iterator[RealDictRow]) -> None:
        asyncio.run(self._tokenize_async(data_chunks))

    async def _tokenize_async(self, data_chunks: Iterator[RealDictRow], keep_in_memory: bool = False) -> None:
        await asyncio.gather(*[self._tokenize_chunk_async(chunk, keep_in_memory) for chunk in data_chunks])

    async def _tokenize_chunk_async(self, chunk: RealDictRow, keep_in_memory: bool = False) -> None:
        insert_query: str = """
        INSERT INTO tokens (clean_metadata_id, token_type, tokens) VALUES ($1, $2, $3)
        ON CONFLICT (clean_metadata_id, token_type) 
        DO UPDATE SET tokens = EXCLUDED.tokens;
        """
        records_to_insert: list[tuple[int, str, list[str]]] = []
        conn = await utils.db_get_conn_async()
        for instance in chunk:
            tokens: list[str] = self._tokenizer(instance)
            table_index = instance.get('id')
            record: tuple[int, str, list[str]] = (table_index, self.token_type, tokens)
            records_to_insert.append(record)
            if keep_in_memory:
                self.tokens.append(tokens)
        async with conn.transaction():
            await conn.executemany(insert_query, records_to_insert)
        await conn.close()

    # TODO: I can compress these indices maps with simple math
    def generate_index(self, keep_in_memory: bool = False) -> None:
        query = "SELECT clean_metadata_id, tokens FROM tokens WHERE token_type = %s;"
        indices_map = []
        token_indices_map = []
        with utils.db_get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (self.token_type,))
                instances = cursor.fetchall()
                for instance in instances:
                    tokens = instance['tokens']
                    indices_map.extend([instance['clean_metadata_id']] * len(tokens))
                    token_indices_map.extend(list(range(len(tokens))))

        if keep_in_memory:
            self.tokens = tokens
            self.token_indices_map = token_indices_map
            self.indices_map = indices_map
        utils.indices_save((indices_map, token_indices_map), self.token_type, self.language)

    def tokenize(self, data_chunks: Iterator[RealDictRow], keep_in_memory: bool = False) -> None:
        # TODO: caching is now done in the database, update this
        if config.ALLOW_CACHING and os.path.exists(self.path):
            logger.info(f"Tokens already cached: {self.path}.")
            return None
        ix: int = 0
        indices: list[int] = []
        with open(self.path, "wb") as f:
            for batch in data_chunks:
                batch_tokens: list[list[str]] = []
                for instance in batch:
                    token: list[str] = self._tokenizer(instance)
                    indices += [ix] * len(token)
                    ix += 1
                    batch_tokens.append(token)
                if keep_in_memory:
                    self.tokens.append(self._tokenizer(instance))
                else:
                    pickle.dump(batch_tokens, f)
        with open(self.indices_map_path, "wb") as f:
            pickle.dump(indices, f)

    def tokens_generator(self) -> Iterator[list[str]]:
        with open(self.path, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    @classmethod
    def token_types(cls) -> list[str]:
        return [attr[10:-7] for attr in dir(cls) if attr.startswith("_generate_") and attr.endswith("_tokens")]

    def _save(self):
        utils.tokens_save(self.tokens, self.token_type, self.language)
        logger.info(f"Tokens generated and saved to {self.path}.")

    def clear_cache(self):
        del self.tokens
        gc.collect()
        self.tokens = []

    def _get_instance_tokens(self, instance: RealDictRow) -> tuple[list[str], list[str], list[str]]:
        tt: list[str] = instance[f"title_tokens_{self.language}"]
        at: list[str] = instance[f"abstract_tokens_{self.language}"]
        kt: list[str] = instance[f"keywords_tokens_{self.language}"]

        if tt is None:
            tt = [""]
        assert at is not None
        assert kt is not None

        return tt, at, kt

    def _generate_paragraph_tokens(self, instance: RealDictRow) -> list[str]:
        tt, at, kt = self._get_instance_tokens(instance)
        paragraph_tokens: list[str] = [". ".join(tt + at + kt)]
        assert len(paragraph_tokens) > 0
        return paragraph_tokens

    def _generate_sentence_tokens(self, instance: RealDictRow) -> list[str]:
        tu, au, ku = self._get_instance_tokens(instance)
        sentence_tokens: list[str] = tu + au + ku
        assert len(sentence_tokens) > 0
        return sentence_tokens

    def _generate_sentence_with_keywords_tokens(self, instance: RealDictRow) -> list[str]:
        tt, at, kt = self._get_instance_tokens(instance)
        sentence_tokens: list[str] = tt + at
        kw_str: str = ". ".join(kt)
        for i in range(len(sentence_tokens)):
            sentence_tokens[i] += f". {kw_str}"

        assert len(sentence_tokens) > 0
        return sentence_tokens

    # def _generate_8gram_tokens(self, instance: RealDictRow) -> list[str]:
    #     n = 8
    #     pt: dict[str, list[str]] = {}
    #     pt['tt'], pt['at'], pt['kt'] = self._get_instance_tokens(instance)
    #     ngrams: dict[str, list[str]] = {
    #         "tt": [],
    #         "at": [],
    #     }
    #     for pta in ('tt', 'at'):
    #         p = pt[pta]
    #         for i in range(len(p)):
    #             tokens: list[str] = self.__generate_ngram_tokens(p[i].split(), n)
    #             if tokens is None or len(tokens) == 0:
    #                 continue
    #             assert isinstance(tokens[0], str)
    #             ngrams[pta].extend(tokens)
    #
    #     tokens: list[str] = ngrams['tt'] + ngrams['at']
    #     if len(tokens) <= 0:
    #         raise ValueError(f"Empty tokens for instance: {instance}")
    #     return tokens
    #
    # def _generate_8gram_with_keywords_tokens(self, instance: RealDictRow) -> list[str]:
    #     n = 8
    #     pt: dict[str, list[str]] = {}
    #     pt['tt'], pt['at'], kt = self._get_instance_tokens(instance)
    #     kw_str = ". ".join(kt)
    #     ngrams: dict[str, list[str]] = {
    #         "tt": [],
    #         "at": [],
    #     }
    #     for pta in ('tt', 'at'):
    #         p = pt[pta]
    #         for i in range(len(p)):
    #             tokens: list[str] = self.__generate_ngram_tokens(p[i].split(), n, kw_str)
    #             if tokens is None or len(tokens) == 0:
    #                 continue
    #             assert isinstance(tokens[0], str)
    #             ngrams[pta].extend(tokens)
    #     tokens: list[str] = ngrams['tt'] + ngrams['at']
    #     assert len(tokens) > 0
    #     return tokens

    @staticmethod
    def __generate_ngram_tokens(
            words: list[str], n: int, kw_tokens: Optional[str] = None
    ) -> list[str]:
        """
        Generates n-gram tokens from a list of words.
        """
        assert isinstance(words, list)
        if len(words) < n:
            return [" ".join(words)]

        ngram_tokens: list[str] = []
        for i in range(len(words) - n + 1):
            ngram: str = " ".join(words[i:i + n])
            if kw_tokens:
                ngram += f". {kw_tokens}"
            ngram_tokens.append(ngram)
        return ngram_tokens


if __name__ == "__main__":
    languages: list[str] = ["pt", "en"]
    for language in languages:
        for token_type in Tokenizer.token_types():
            tokenizer = Tokenizer(token_type, language)
            data: Iterator[RealDictRow] = utils.tokenized_metadata_generator()
            # tokenizer.tokenize_async(data)
            tokenizer.generate_index()
