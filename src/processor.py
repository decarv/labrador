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

from psycopg2.extras import RealDictRow

from models.metadata import Metadata
from utils import log, metadata_read, tokenized_metadata_insert


class Processor:
    """
    Processing component for text.
    Implementation will encompass all text processing steps, such as:
     a) standardization;
     b) tokenization;
     c) spell check for queries.
    """

    @staticmethod
    def _split_by_delimiters(string, delimiters: list[str] | str) -> list[str]:
        if isinstance(delimiters, list):
            pattern = r"|".join(delimiters)
        else:
            pattern = r"{}".format(delimiters)
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

    @log
    def process_metadata(self):
        metadata_objects: list[RealDictRow] = metadata_read()
        for entry in metadata_objects:
            url: str = entry['url']
            title_pt_tokens: list[str] = self._clean_string_list(self._split_by_delimiters(entry['title_pt'], ['.']))
            abstract_pt_tokens: list[str] = self._clean_string_list(
                self._split_by_delimiters(entry['abstract_pt'], ['.']))
            keywords_pt_tokens: list[str] = self._split_keywords(entry['keywords_pt'])
            title_en_tokens: list[str] = self._clean_string_list(self._split_by_delimiters(entry['title_en'], ['.']))
            abstract_en_tokens: list[str] = self._clean_string_list(
                self._split_by_delimiters(entry['abstract_en'], ['.']))
            keywords_en_tokens: list[str] = self._split_keywords(entry['keywords_en'])

            tokenized_metadata_insert(
                url=url,
                title_pt_tokens=title_pt_tokens,
                abstract_pt_tokens=abstract_pt_tokens,
                keywords_pt_tokens=keywords_pt_tokens,
                title_en_tokens=title_en_tokens,
                abstract_en_tokens=abstract_en_tokens,
                keywords_en_tokens=keywords_en_tokens
            )


if __name__ == '__main__':
    processor = Processor()
    processor.process_metadata()
