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
import time
from typing import Iterator

import pandas as pd
from psycopg2.extras import RealDictRow

from labrador.util import log, database

logger = log.configure_logger(__file__)


class Processor:
    """
    Processing component for text.
    Encompass text processing steps, such as standardization & cleaning.

    Notes for paper:
        Problems that I had to solve:
            (a) unwanted strings such as "Não disponível";
            (b) abstracts, titles and keywords empty strings;
            (c) duplicate abstracts, titles and keywords;
            (d) non-uniform way to list keywords (separated by ';' or '-' or all in lower-case);

    TODO: Implement stemming and lemmatization.
        - https://www.nltk.org/book/ch03.html
        - https://www.nltk.org/book/ch05.html
        - Remove prefix from urls
    """
    def __init__(self, language="pt"):
        # TODO
        self.language = language

    def loop(self):
        batch_size: int = 64
        while True:
            batch_generator: Iterator[list[RealDictRow]] = database.raw_data_batch_generator(batch_size=batch_size)
            for batch in batch_generator:
                df: pd.DataFrame = pd.DataFrame([dict(record) for record in batch])
                records_to_insert: list[dict] = self._clean_dataframe(df)
                if len(records_to_insert) > 0:
                    database.clean_data_batch_insert(records_to_insert)

            logger.info("No records to process. Sleeping for 5 minutes.")
            time.sleep(300)
            continue

    def _clean_dataframe(self, df: pd.DataFrame) -> list[dict]:
        # Remove rows where title_pt or abstract_pt is NaN.
        df.dropna(subset=['title_pt', 'abstract_pt'], inplace=True)

        for col in ['abstract_pt', 'keywords_pt', 'title_pt']:
            df[col] = df[col].apply(lambda x: "" if x is None else x)

        # Replace unwanted strings in columns
        unwanted_strings: set = {
            'não disponível', 'Não disponível', 'Não consta', 'Não disponível.',
            'Não consta resumo na publicação.', '-', 'Não consta.', 'Sem resumo',
            'Não possui resumo.', 'Not available', 'Resumo',
            'Não disponível pelo autor.', 'Não informado pelo autor.', '',
            'Sem Resumo', 'Não Consta Resumo na Publicação',
            'Não fornecido pelo autor.', 'Não consta resumo na publicação',
            'Sem resumo.', 'não consta.', 'not available', 'não há resumo',
            'Sem resumo em português', 'Sem resumo em português.', 'não possui',
            'Abstract not available.', 'Não possui resumo', 'Não possui resumo.',
        }
        for col in ['abstract_pt', 'keywords_pt', 'title_pt']:
            df[col] = df[col].apply(lambda x: "" if x.strip() in unwanted_strings else x)

        # Remove rows where abstract_pt or title_pt is an empty string.
        df = df[df['abstract_pt'].str.strip() != ""]
        df.reset_index(drop=True, inplace=True)
        df = df[df['title_pt'].str.strip() != ""]
        df.reset_index(drop=True, inplace=True)

        # Drop duplicates based on title and abstract
        df.drop_duplicates(subset=['abstract_pt'], keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.drop_duplicates(subset=['title_pt'], keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Clean unwanted substring from author
        df['author'] = df['author'].str.replace("(Catálogo USP)", "")

        # Clean " from titles
        df['title_pt'] = df['title_pt'].str.strip("\"")

        df['keywords_pt'] = df['keywords_pt'].apply(self.clean_keywords)

        df['url_path_suffix'] = df['url'].apply(lambda x: self.extract_path_suffix(x))

        df.drop(columns=['cleaned'], inplace=True)
        df.rename(columns={'id': 'raw_data_id'}, inplace=True)

        return df.to_dict(orient='records')

    @staticmethod
    def clean_keywords(text: str) -> str:
        """
        Cleans a string of keywords.
        TODO:
            - remove stopwords
            - improve bi-words recognition. e.g. "São Paulo" is currently split with a semicolon
        """

        masked_text: str = Processor.mask_problematic_punctuation(text)
        split_text: list[str] = re.split(r"[;.,-]", masked_text)
        if len(split_text) == 1:
            split_text = split_text[0].split(" ")
            keywords: list[str] = []
            keyword: str = ""
            for word in split_text:
                if word == "":
                    continue
                if word[0].isupper():
                    if keyword != "":
                        keywords.append(keyword.strip())
                    keyword = word
                else:
                    keyword += " " + word
            keywords.append(keyword)
        else:
            keywords: list[str] = split_text

        keywords = Processor.clean_string_list(keywords)
        masked_keywords: str = "; ".join(keywords)
        unmasked_clean_keywords: str = Processor.unmask_problematic_punctuation(masked_keywords)

        return unmasked_clean_keywords

    @staticmethod
    def mask_problematic_punctuation(text: str) -> str:
        masked_text: str = text
        masked_text = re.sub(r"([0-9]+)\.([0-9]+)", r"\1[DOT]\2", masked_text)

        # Mask dots in abbreviations
        abbreviation_patterns = r"\b(Dr|Dra|Sr|Sra|[A-Za-z]|et al|var)\."
        masked_text = re.sub(abbreviation_patterns, r"\1[DOT]", masked_text)

        # Mask commas in compound names like 2,4,6-trinitrotoluene
        masked_text = re.sub(
            r"[0-9A-Za-z]+,?[0-9A-Z]*?-[a-zA-Z]+",
            lambda x: x.group().replace(",", "[COMMA]").replace("-", "[DASH]"),
            masked_text
        )

        return masked_text

    @staticmethod
    def unmask_problematic_punctuation(text: str) -> str:
        """
        Replaces masked punctuation with the original punctuation.
        TODO:
            - Consider not unmasking before adding to the database, since I will unmask it in the tokenizer.
        """
        unmasked_text: str = text.replace("[DOT]", ".")
        unmasked_text = unmasked_text.replace("[COMMA]", ",")
        unmasked_text = unmasked_text.replace("[DASH]", "-")
        return unmasked_text

    @staticmethod
    def clean_string_list(strings: list[str]) -> list[str]:
        strings = [s.strip(" -.") for s in strings]  # Remove leading and trailing whitespaces
        strings = [s for s in strings if s != '']  # Remove empty strings

        strings = [re.sub(r"\s+", " ", s) for s in strings]  # Remove multiple whitespaces
        zero_width_pattern = r"\u00AD|\u200B|\u200D|\u200E|\u200F|\u202C|\uFEFF|\u200C|\u2060"
        strings = [re.sub(zero_width_pattern, "", s) for s in strings]  # Remove zero-width characters
        return strings

    @staticmethod
    def remove_lang_from_url(url):
        return re.sub(r'/\?&lang.*$', '', url)

    @staticmethod
    def extract_path_suffix(url):
        pattern = r"disponiveis\/(.*\/td.*\d)"
        match = re.search(pattern, url)
        path_suffix = match.group()
        return path_suffix


if __name__ == '__main__':
    processor = Processor()
    processor.loop()