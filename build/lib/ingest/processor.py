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
import json
import asyncio

import asyncpg
import pandas as pd

from util import log
import config

logger = log.configure_logger(__file__)
log = log.log(logger)


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
    """
    def __init__(self):
        pass

    async def process_table(self):
        insert_query = """
            INSERT INTO data (
                url, 
                doi,
                type,
                author,
                institute,
                knowledge_area,
                committee,
                title_pt, 
                title_en,
                keywords_pt, 
                keywords_en,
                abstract_pt, 
                abstract_en, 
                publish_date,
                raw_data_id
            ) 
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, to_date($14, 'YYYY-MM-DD'), $15)
            ON CONFLICT (raw_data_id)
            DO UPDATE SET 
                keywords_pt = EXCLUDED.keywords_pt;
        """

        pool = await asyncpg.create_pool(
            database=config.POSTGRESQL_DB_NAME,
            user=config.POSTGRESQL_DB_USER,
            password=config.POSTGRESQL_DB_PASSWORD,
            host=config.POSTGRESQL_DB_HOST,
            port=config.POSTGRESQL_DB_PORT,
        )
        async with pool.acquire() as conn:
            await conn.set_type_codec(
                'json',
                encoder=json.dumps,
                decoder=json.loads,
                schema='pg_catalog'
            )
            metadata_records = await conn.fetch(f"SELECT * FROM metadata ORDER BY id;")
            metadata_df: pd.DataFrame = pd.DataFrame([dict(record) for record in metadata_records])
            metadata_df = self._clean_dataframe(metadata_df)
            records_to_insert = [row[1] for row in metadata_df.iterrows()]
            async with conn.transaction():
                await conn.executemany(insert_query, records_to_insert)

    @log
    def _clean_dataframe(self, df: pd.DataFrame):
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

        df['keywords_pt'] = df['keywords_pt'].apply(self.clean_keywords)

        return df

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


if __name__ == '__main__':
    processor = Processor()
    asyncio.run(processor.process_table())
