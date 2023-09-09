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
import json
import re
import asyncpg
import asyncio
import pandas as pd
import psycopg2.extras

import utils
import config

logger = utils.configure_logger(__name__)
log = utils.log(logger)


class Processor:
    """
    Processing component for text.
    Encompass text processing steps, such as standardization & cleaning.
    """
    def __init__(self):
        pass

    async def process_table(self):
        insert_query = """
            INSERT INTO clean_metadata (
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
                metadata_id
            ) 
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, to_date($14, 'YYYY-MM-DD'), $15)
            ON CONFLICT (metadata_id)
            DO UPDATE SET 
                title_pt = EXCLUDED.title_pt, 
                abstract_pt = EXCLUDED.abstract_pt, 
                keywords_pt = EXCLUDED.keywords_pt, 
                title_en = EXCLUDED.title_en, 
                abstract_en = EXCLUDED.abstract_en, 
                keywords_en = EXCLUDED.keywords_en,
                publish_date = EXCLUDED.publish_date,
                metadata_id = EXCLUDED.metadata_id;
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

        masked_text: str = utils.mask_problematic_punctuation(text)
        split_text: list[str] = re.split(r"([;.,])", masked_text)
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

        keywords = utils.clean_string_list(keywords)
        masked_keywords: str = "; ".join(keywords)
        unmasked_clean_keywords: str = utils.unmask_problematic_punctuation(masked_keywords)

        return unmasked_clean_keywords


if __name__ == '__main__':
    processor = Processor()
    asyncio.run(processor.process_table())
