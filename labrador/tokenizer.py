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
import utils

logger = utils.configure_logger(__name__)
log = utils.log(logger)


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

    def tokenize(self, data_chunks: Iterator[RealDictRow], keep_in_memory: bool = False) -> None:
        for chunk in data_chunks:
            self._tokenize_chunk(chunk, keep_in_memory)
        # asyncio.run(self._tokenize_async(data_chunks, keep_in_memory))

    def _tokenize_chunk(self, chunk, keep_in_memory=False):
        insert_query = """
        INSERT INTO clean_metadata_tokens (clean_metadata_id, token, token_type, language, unique_hash)
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
                for token in tokens:
                    unique_str = f"{ref_id}{token}{self.token_type}{self.language}"
                    unique_hash = hashlib.md5(unique_str.encode()).hexdigest()
                    record = (ref_id, token, self.token_type, self.language, unique_hash)

                    if keep_in_memory:
                        self.tokens.append(token)

                    try:
                        cursor.execute(insert_query, record)
                        conn.commit()
                    except Exception as e:
                        logger.error(f"Error inserting metadata_id {ref_id} into database: {e}")
        finally:
            cursor.close()
            conn.close()

    async def _tokenize_async(self, data_chunks: Iterator[RealDictRow], keep_in_memory: bool) -> None:
        await asyncio.gather(*[self._tokenize_chunk_async(chunk, keep_in_memory) for chunk in data_chunks])

    @log
    async def _tokenize_chunk_async(self, chunk: RealDictRow, keep_in_memory: bool) -> None:
        insert_query = """
        INSERT INTO clean_metadata_tokens (clean_metadata_id, token, token_type, language, unique_hash)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (unique_hash) DO NOTHING;
        """
        # records_to_insert: list[tuple[int, str, str, str, str]] = []
        conn = await asyncpg.connect(
            database=config.POSTGRESQL_DB_NAME,
            user=config.POSTGRESQL_DB_USER,
            password=config.POSTGRESQL_DB_PASSWORD,
            host=config.POSTGRESQL_DB_HOST,
            port=config.POSTGRESQL_DB_PORT
        )

        for instance in chunk:
            ref_id: int = instance.get('id')
            tokens: list[str] = self._tokenizer(instance)
            for token in tokens:
                unique_str = f"{ref_id}{token}{self.token_type}{self.language}"
                unique_hash: str = hashlib.md5(unique_str.encode()).hexdigest()
                record: tuple[int, str, str, str, str] = (ref_id, token, self.token_type,
                                                          self.language, unique_hash)
                # records_to_insert.append(record)
                if keep_in_memory:
                    self.tokens.append(token)
                try:
                    await conn.execute(insert_query, record)
                except Exception as e:
                    logger.error(f"Error inserting metadata_id {ref_id} into database: {e}")
        # async with conn.transaction():
        #     await conn.executemany(insert_query, records_to_insert)
        conn.close()

    def tokens_generator(self) -> Iterator[list[str]]:
        raise NotImplementedError

    @classmethod
    def token_types(cls) -> list[str]:
        return [attr[10:-7] for attr in dir(cls) if attr.startswith("_generate_") and attr.endswith("_tokens")]

    def _get_data_to_tokenize(self, instance: RealDictRow) -> tuple[str, str, str]:
        title: str = instance[f"title_{self.language}"]
        assert title is not None

        abstract: str = instance[f"abstract_{self.language}"]
        assert abstract is not None

        keywords: str = instance[f"keywords_{self.language}"]
        if keywords is None:
            keywords = ""

        return title, abstract, keywords

    def _generate_paragraph_tokens(self, instance: RealDictRow) -> list[str]:
        title, abstract, keywords = self._get_data_to_tokenize(instance)
        paragraph_tokens: list[str] = [title + ". "+ abstract + keywords]
        assert len(paragraph_tokens) > 0
        return paragraph_tokens

    def _generate_sentence_tokens(self, instance: RealDictRow) -> list[str]:
        """
        Each token is a sentence.
        """
        title, abstract, keywords = self._get_data_to_tokenize(instance)
        sentence_tokens: list[str] = [title] + self.tokenize_sentences(abstract) + self.tokenize_keywords(keywords)
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
        masked_text: str = utils.mask_problematic_punctuation(text)

        # Split by delimiters
        sentences: list[str] = re.split(r"([.!?])\s", masked_text)

        # Unmask the special cases by replacing the sentinel value back to dots
        unmasked_sentences: list[str] = []
        for sentence in sentences:
            unmasked_sentences.append(utils.unmask_problematic_punctuation(sentence))
        unmasked_sentences = utils.clean_string_list(unmasked_sentences)

        return unmasked_sentences


if __name__ == "__main__":
    languages: list[str] = ["pt"]
    for language in languages:
        for token_type in Tokenizer.token_types():
            if token_type == "paragraph":
                continue
            tokenizer = Tokenizer(token_type, language)
            data: Iterator[RealDictRow] = utils.metadata_generator()
            tokenizer.tokenize(data)


"""
CREATE TABLE clean_metadata_tokens (
    id SERIAL PRIMARY KEY,
    clean_metadata_id integer NOT NULL,
    token text,
    token_type text,
    language text,
    unique_hash varchar(32) NOT NULL
    );
"""
