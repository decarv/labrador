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
import hashlib
import re
from typing import Optional, Callable, Iterator

import asyncpg
import psycopg2
from psycopg2.extras import RealDictRow

import config
import util.log
from ingest.processor import Processor
from util import utils, database
import time

logger = util.log.configure_logger(__file__)
log = util.log.log(logger)


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

        self.tokens: Optional[list[str]] = None
        self.indices_map: Optional[list[int]] = None
        self.token_indices_map: Optional[list[int]] = None

        self.pool = None
        # asyncio.run(self.initialize())
    
    async def initialize(self):
        self.pool = await asyncpg.create_pool(
            database=config.POSTGRESQL_DB_NAME,
            user=config.POSTGRESQL_DB_USER,
            password=config.POSTGRESQL_DB_PASSWORD,
            host=config.POSTGRESQL_DB_HOST,
            port=config.POSTGRESQL_DB_PORT,
        )

    @log
    def tokenize(self, data_chunks: Iterator[list[RealDictRow]]) -> None:
        for batch in data_chunks:
            self._tokenize_batch(batch)

    @log
    def _tokenize_batch(self, chunk: list[RealDictRow], keep_in_memory=False):
        insert_query = """
        INSERT INTO tokens (data_id, token, token_type, language, unique_hash)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (unique_hash) DO NOTHING;
        """

        conn = psycopg2.connect(
            dbname=config.POSTGRESQL_DB_NAME,
            user=config.POSTGRESQL_DB_USER,
            password=config.POSTGRESQL_DB_PASSWORD,
            host=config.POSTGRESQL_DB_HOST,
            port=config.POSTGRESQL_DB_PORT
        )
        cursor = conn.cursor()

        try:
            for instance in chunk:
                ref_id = instance.get('id')
                tokens = self._tokenizer(instance)
                records = []
                for token in tokens:
                    unique_str = f"{ref_id}{token}{self.token_type}{self.language}"
                    unique_hash = hashlib.md5(unique_str.encode()).hexdigest()
                    record = (ref_id, token, self.token_type, self.language, unique_hash)
                    records.append(record)

                try:
                    cursor.executemany(insert_query, records)
                    conn.commit()
                except Exception as e:
                    logger.error(f"Error inserting metadata_id {ref_id} into database: {e}")
        finally:
            cursor.close()
            conn.close()

    @log
    def tokenize_async(self, doc_chunks: Iterator[list[RealDictRow]]) -> None:
        asyncio.run(self._tokenize_async(doc_chunks))

    async def _tokenize_async(self, doc_chunks: Iterator[list[RealDictRow]]) -> None:
        await asyncio.gather(*[self._tokenize_chunk_async(chunk) for chunk in doc_chunks])

    @log
    async def _tokenize_chunk_async(self, chunk: list[RealDictRow]) -> None:
        insert_query = """
        INSERT INTO tokens (data_id, token, token_type, language, unique_hash)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (unique_hash) DO NOTHING;
        """

        records: list[tuple[int, str, str, str, str]] = []
        for instance in chunk:
            doc_id: int = instance.get('id')
            tokens: list[str] = self._tokenizer(instance)
            for token in tokens:
                unique_str = f"{doc_id}{token}{self.token_type}{self.language}"
                unique_hash: str = hashlib.md5(unique_str.encode()).hexdigest()
                record: tuple[int, str, str, str, str] = (doc_id, token, self.token_type,
                                                          self.language, unique_hash)
                records.append(record)

        conn = None
        try:
            conn = await database.get_async_connection()
            async with conn.cursor() as cursor:
                for record in records:
                    await cursor.execute(insert_query, record)
        except Exception as e:
            logger.error(f"Error inserting records into database: {e}: {records[0]}")
            print(f"Error inserting records into database: {e}: {records[0]}")
            raise e
        finally:
            await cursor.close()
            await conn.close()

    def tokens_generator(self) -> Iterator[list[str]]:
        raise NotImplementedError

    @classmethod
    def token_types(cls) -> list[str]:
        return [attr[10:-7] for attr in dir(cls) if attr.startswith("_generate_") and attr.endswith("_tokens")]

    def _get_data_to_tokenize(self, instance: RealDictRow) -> tuple[str, str, str]:
        try:
            title: str = instance[f"title_{self.language}"]
            assert title is not None

            abstract: str = instance[f"abstract_{self.language}"]
            assert abstract is not None

            keywords: str = instance[f"keywords_{self.language}"]
            if keywords is None:
                keywords = ""
        except KeyError:
            logger.error(f"Error getting data to tokenize from instance: {instance}")
            database.errors_insert(f"Error getting data to tokenize from instance: {instance}")
            return "", "", ""

        return title, abstract, keywords

    def _generate_paragraph_tokens(self, instance: RealDictRow) -> list[str]:
        title, abstract, keywords = self._get_data_to_tokenize(instance)
        paragraph_tokens: list[str] = [title + ". " + abstract + keywords]
        assert len(paragraph_tokens) > 0
        return paragraph_tokens

    def _generate_sentence_tokens(self, instance: RealDictRow) -> list[str]:
        """
        Each token is a sentence.
        """
        title, abstract, keywords = self._get_data_to_tokenize(instance)
        sentence_tokens: list[str] = [title] + self.tokenize_sentences(abstract) + [keywords]
        assert len(sentence_tokens) > 0
        return sentence_tokens

    def _generate_sentence_with_keywords_tokens(self, instance: RealDictRow) -> list[str]:
        title, abstract, keywords = self._get_data_to_tokenize(instance)
        sentence_tokens: list[str] = [title] + self.tokenize_sentences(abstract)
        for i in range(len(sentence_tokens)):
            sentence_tokens[i] += f". {keywords}"
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

    @staticmethod
    def tokenize_keywords(text) -> list[str]:
        """
        Splits a string of keywords into a list of keywords. Assumes that the keywords are separated by semicolon
        followed by space: i.e., "; ".
        """
        return text.split("; ")

    @staticmethod
    def tokenize_sentences(text) -> list[str]:
        # Mask problematic dots with a sentinel value
        masked_text: str = Processor.mask_problematic_punctuation(text)

        # Split by delimiters
        sentences: list[str] = re.split(r"[.!?]\s", masked_text)

        # Unmask the special cases by replacing the sentinel value back to dots
        unmasked_sentences: list[str] = []
        for sentence in sentences:
            unmasked_sentences.append(Processor.unmask_problematic_punctuation(sentence))
        unmasked_sentences = Processor.clean_string_list(unmasked_sentences)

        return unmasked_sentences


if __name__ == "__main__":
    language: str = "pt"
    token_type = "sentence_with_keywords"
    tokenizer = Tokenizer(token_type, language)
    while True:
        documents: Iterator[list[RealDictRow]] = database.documents_chunk_generator()
        tokenizer.tokenize(documents)
        time.sleep(300)
