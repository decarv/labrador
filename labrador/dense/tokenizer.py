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

import enum
import hashlib
import re
from typing import Optional, Callable, Iterator

from labrador.util.log import configure_logger, logger
from labrador.util.database import Database
from labrador.dense.processor import Processor
import time

configure_logger(__file__)


class TokenTypes(enum.Enum):
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    SENTENCE_WITH_KEYWORDS = "sentence_with_keywords"
    # EIGHTGRAM = "8gram"
    # EIGHTGRAM_WITH_KEYWORDS = "8gram_with_keywords"


class Tokenizer:
    class TokenizerError(Exception):
        pass

    def __init__(self, database: Database, language: str = "pt", token_type="sentence_with_keywords"):
        self._database: Database = database
        self.token_type = token_type
        self.language: str = language

    def loop(self, id_range: tuple[int, int] = None):
        """
        This function is responsible for the tokenization pipeline. It fetches documents from the database and
        tokenizes them using the Tokenizer class. The tokenization process is done in a loop, so that it can be
        repeated indefinitely.
        """

        if id_range is None:
            id_range = (1, self._database.select("""SELECT id FROM documents ORDER BY id DESC LIMIT 1;""")[0]['id'])

        while True:
            batches: Iterator[list] = self._database.batch_generator(
                """
                SELECT d.id as doc_id, title_pt as title, abstract_pt as abstract, keywords_pt as keywords
                FROM documents as d
                WHERE d.id BETWEEN %s AND %s
                AND d.id NOT IN (SELECT DISTINCT d.id as doc_id
                    FROM documents as d
                    JOIN tokens AS t ON d.id = t.doc_id
                    WHERE token_type = %s);
                """,
                (*id_range, self.token_type,),
                batch_size=128
            )

            for batch in batches:
                records = self._tokenize_batch(self.token_type, batch)
                try:
                    self._database.insert_many(
                            """INSERT INTO tokens (doc_id, token, token_type, language, unique_hash)
                               VALUES (%s, %s, %s, %s, %s)
                               ON CONFLICT (unique_hash) DO NOTHING;""",
                            records
                    )
                except Database.DatabaseError as e:
                    logger.error(f"Error inserting records into database: {e}")

            time.sleep(3600)

    def _tokenize_batch(self, token_type: str, batch: list) -> list[tuple[int, str, str, str, str]]:
        tokenizer_function: Callable = getattr(self, f"_generate_{token_type}_tokens")
        records = []
        for document_record in batch:
            doc_id = document_record.get('doc_id')
            tokens = tokenizer_function(document_record)
            for token in tokens:
                unique_str = f"{token}{token_type}{self.language}"
                unique_hash = hashlib.md5(unique_str.encode()).hexdigest()
                record: tuple[int, str, str, str, str] = (doc_id, token, token_type, self.language, unique_hash)
                records.append(record)

        return records

    # @log
    # def tokenize_async(self, doc_chunks: Iterator[list[RealDictRow]]) -> None:
    #     asyncio.run(self._tokenize_async(doc_chunks))
    #
    # async def _tokenize_async(self, doc_chunks: Iterator[list[RealDictRow]]) -> None:
    #     await asyncio.gather(*[self._tokenize_chunk_async(chunk) for chunk in doc_chunks])
    #
    # @log
    # async def _tokenize_chunk_async(self, chunk: list[RealDictRow]) -> None:
    #     insert_query = """
    #     INSERT INTO tokens (data_id, token, token_type, language, unique_hash)
    #     VALUES (%s, %s, %s, %s, %s)
    #     ON CONFLICT (unique_hash) DO NOTHING;
    #     """
    #
    #     records: list[tuple[int, str, str, str, str]] = []
    #     for instance in chunk:
    #         doc_id: int = instance.get('id')
    #         tokens: list[str] = self._tokenizer(instance)
    #         for token in tokens:
    #             unique_str = f"{doc_id}{token}{self.token_type}{self.language}"
    #             unique_hash: str = hashlib.md5(unique_str.encode()).hexdigest()
    #             record: tuple[int, str, str, str, str] = (doc_id, token, self.token_type,
    #                                                       self.language, unique_hash)
    #             records.append(record)
    #
    #     conn = None
    #     try:
    #         conn = await database.get_async_connection()
    #         async with conn.cursor() as cursor:
    #             for record in records:
    #                 await cursor.execute(insert_query, record)
    #     except Exception as e:
    #         logger.error(f"Error inserting records into database: {e}: {records[0]}")
    #         print(f"Error inserting records into database: {e}: {records[0]}")
    #         raise e
    #     finally:
    #         await cursor.close()
    #         await conn.close()

    @classmethod
    def token_types(cls) -> list[str]:
        return [attr[10:-7] for attr in dir(cls) if attr.startswith("_generate_") and attr.endswith("_tokens")]

    def _generate_paragraph_tokens(self, document: dict) -> list[str]:
        paragraph_tokens: list[str] = [document['title'] + ". " + document['abstract'] + document['keywords']]
        assert len(paragraph_tokens) > 0
        return paragraph_tokens

    def _generate_sentence_tokens(self, document: dict) -> list[str]:
        sentence_tokens: list[str] = [document['title']] + self.tokenize_sentences(document['abstract']) + [document['keywords']]
        assert len(sentence_tokens) > 0
        return sentence_tokens

    def _generate_sentence_with_keywords_tokens(self, document: dict) -> list[str]:
        sentence_tokens: list[str] = [document['title']] + self.tokenize_sentences(document['abstract'])
        for i in range(len(sentence_tokens)):
            sentence_tokens[i] += f". {document['keywords']}"
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


