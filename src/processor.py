"""
processor

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

import re
from typing import Optional
import asyncpg
import asyncio
from psycopg2.extras import RealDictRow

from config import POSTGRESQL_DB_NAME, POSTGRESQL_DB_USER, POSTGRESQL_DB_PASSWORD, POSTGRESQL_DB_HOST, \
    POSTGRESQL_DB_PORT
from models.metadata import Metadata
from utils import log, metadata_read, tokenized_metadata_insert, configure_logger

logger = configure_logger(__name__)


class Processor:
    """
    Processing component for text.
    Implementation will encompass all text processing steps, such as:
     a) standardization & cleaning;
        TODO:
            - limpar 'Not available' e 'Não disponível'.
     b) tokenization;
     c) spell check for queries.
    """

    @staticmethod
    def _split_by_delimiters(string, delimiters: list[str] | str) -> list[str]:
        if isinstance(delimiters, list):
            pattern = r"|".join([re.escape(delimiter) for delimiter in delimiters])
        else:
            pattern = re.escape(delimiters)
        return re.split(pattern, string)

    @staticmethod
    def _split_keywords(string: str) -> list[str]:
        keywords: list[str] = []
        keyword = ""
        for word in string.split():
            if word[0].isupper():
                if keyword != "":
                    keywords.append(keyword.strip())
                keyword = word
            else:
                keyword += " " + word
        keywords.append(keyword)
        return keywords

    @staticmethod
    def _clean_string_list(strings: list[str]) -> list[str]:
        strings = [s.strip() for s in strings]  # Remove leading and trailing whitespaces
        strings = [s for s in strings if s != '']  # Remove empty strings
        return strings
    
    async def process_entry(self, entry):
        metadata_url: str = entry['url']
        title_tokens_pt: Optional[list[str]] = None
        abstract_tokens_pt: Optional[list[str]] = None
        keywords_tokens_pt: Optional[list[str]] = None
        title_tokens_en: Optional[list[str]] = None
        abstract_tokens_en: Optional[list[str]] = None
        keywords_tokens_en: Optional[list[str]] = None

        if entry['title_pt'] is not None:
            title_tokens_pt = self._clean_string_list(

                entry['title_pt'].split('.')
            )
        if entry['abstract_pt'] is not None:
            abstract_tokens_pt = self._clean_string_list(
                entry['abstract_pt'].split('.')
            )
        if entry['keywords_pt'] is not None:
            keywords_tokens_pt = self._split_keywords(entry['keywords_pt'])
        if entry['title_en'] is not None:
            title_tokens_en = self._clean_string_list(
                entry['title_en'].split('.')
            )
        if entry['abstract_en'] is not None:
            abstract_tokens_en = self._clean_string_list(
                entry['abstract_en'].split('.')
            )
        if entry['keywords_en'] is not None:
            keywords_tokens_en = self._split_keywords(entry['keywords_en'])

        return (
            metadata_url,
            title_tokens_pt,
            abstract_tokens_pt,
            keywords_tokens_pt,
            title_tokens_en,
            abstract_tokens_en,
            keywords_tokens_en
        )

    async def process_metadata(self):
        conn = await asyncpg.connect(
            database=POSTGRESQL_DB_NAME,
            user=POSTGRESQL_DB_USER,
            password=POSTGRESQL_DB_PASSWORD,
            host=POSTGRESQL_DB_HOST,
            port=POSTGRESQL_DB_PORT
        )

        metadata_objects = await conn.fetch(f"SELECT * FROM metadata;")

        insert_query = """
            INSERT INTO tokenized_metadata (metadata_url, title_tokens_pt, abstract_tokens_pt, 
            keywords_tokens_pt, title_tokens_en, abstract_tokens_en, keywords_tokens_en) 
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (metadata_url) DO NOTHING;
            """

        # Using asyncio.gather to process entries concurrently
        records_to_insert = await asyncio.gather(*(self.process_entry(entry) for entry in metadata_objects))

        async with conn.transaction():
            await conn.executemany(insert_query, records_to_insert)


if __name__ == '__main__':
    processor = Processor()
    asyncio.run(processor.process_metadata())