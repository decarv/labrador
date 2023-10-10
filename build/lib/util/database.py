"""
database

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

import datetime
from typing import Iterator, Optional

import torch
import asyncpg
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictRow, RealDictCursor

import config
from config import POSTGRESQL_DB_NAME, POSTGRESQL_DB_USER, POSTGRESQL_DB_PASSWORD, POSTGRESQL_DB_HOST, \
    POSTGRESQL_DB_PORT
from ingest.models import Webpage, Metadata

conn_pool: Optional[pool.SimpleConnectionPool] = None


def initialize_conn_pool(minconn: int, maxconn: int):
    global conn_pool
    conn_pool = pool.SimpleConnectionPool(
        minconn,
        maxconn,
        host=config.POSTGRESQL_DB_HOST,
        database=config.POSTGRESQL_DB_NAME,
        port=config.POSTGRESQL_DB_PORT,
        user=config.POSTGRESQL_DB_USER,
        password=config.POSTGRESQL_DB_PASSWORD
    )


def get_conn():
    global conn_pool
    if conn_pool is None:
        raise Exception("Connection pool is not initialized.")
    return conn_pool.getconn()


def release_conn(conn):
    global conn_pool
    if conn_pool is not None:
        conn_pool.putconn(conn)


def table_instance_exists(table: str, column: str, value: str) -> Optional[tuple]:
    conn = get_conn()
    cursor = conn.cursor()
    query = f"SELECT * FROM {table} WHERE {column} = %s;"
    cursor.execute(query, (value,))
    result = cursor.fetchone()
    release_conn(conn)
    return result


def webpages_instance_exists(url: str) -> bool:
    return table_instance_exists("webpages", "url", url) is not None


def metadata_instance_exists(url: str) -> bool:
    return table_instance_exists("metadata", "url", url) is not None


def webpages_crawled_update(webpage: Webpage) -> None:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("UPDATE webpages SET is_crawled = 1 WHERE url = %s;", (webpage.url,))
    conn.commit()
    release_conn(conn)


def webpages_read(iterator_size: int = 4096) -> Iterator[list[RealDictRow]]:
    conn = get_conn()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM webpages;")
    try:
        while True:
            instances = cursor.fetchmany(iterator_size)
            if not instances:
                break
            yield instances
    finally:
        cursor.close()
        release_conn(conn)


def webpages_empty() -> bool:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM webpages;")
    result = cursor.fetchone()
    release_conn(conn)
    return result[0] == 0


def webpages_insert(webpage: Webpage) -> None:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO webpages VALUES (%s, %s, %s, %s)",
        (webpage.url, webpage.is_metadata, webpage.is_document, 0)  # crawled = 0
    )
    conn.commit()
    release_conn(conn)


def metadata_insert(metadata: Metadata) -> None:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO raw_data (
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
            publish_date
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        metadata.url,
        metadata.doi,
        metadata.type,
        metadata.author,
        metadata.institute,
        metadata.knowledge_area,
        metadata.committee,
        metadata.title_pt,
        metadata.title_en,
        metadata.keywords_pt,
        metadata.keywords_en,
        metadata.abstract_pt,
        metadata.abstract_en,
        metadata.publish_date
    ))
    conn.commit()
    release_conn(conn)


def connection():
    return psycopg2.connect(
        dbname=POSTGRESQL_DB_NAME,
        user=POSTGRESQL_DB_USER,
        password=POSTGRESQL_DB_PASSWORD,
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT
    )


def metadata_read() -> list[RealDictRow]:
    with psycopg2.connect(
            dbname=POSTGRESQL_DB_NAME,
            user=POSTGRESQL_DB_USER,
            password=POSTGRESQL_DB_PASSWORD,
            host=POSTGRESQL_DB_HOST,
            port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"SELECT * FROM clean_metadata ORDER BY id;")
            instances = cursor.fetchall()
    return instances


def table_chunk_generator(table_name: str, chunk_size: int = 4096) -> Iterator[list[RealDictRow]]:
    with psycopg2.connect(
        dbname=POSTGRESQL_DB_NAME,
        user=POSTGRESQL_DB_USER,
        password=POSTGRESQL_DB_PASSWORD,
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"SELECT * FROM {table_name} order by id;")
            while True:
                instances = cursor.fetchmany(chunk_size)
                if not instances:
                    break
                yield instances


def embeddings_exists(token_id: int, model_name: str) -> bool:
    select_query: str = """SELECT id 
                             FROM embeddings 
                            WHERE token_id = %s AND model_name = %s;"""

    with psycopg2.connect(
        dbname=POSTGRESQL_DB_NAME,
        user=POSTGRESQL_DB_USER,
        password=POSTGRESQL_DB_PASSWORD,
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(select_query, (token_id, model_name,))
            instances = cursor.fetchall()
    return len(instances) > 0


def embeddings_insert(token_ids: list[int], embeddings: torch.tensor, model_name: str) -> None:
    insert_query: str = """INSERT INTO embeddings (token_id, embeddings, model_name) 
                                VALUES (%s, %s, %s) 
                           ON CONFLICT ON CONSTRAINT embeddings_token_id_model_name_key DO NOTHING;"""
    insert_data: list[tuple[int, list[float], str]] = [
        (tid, e, model_name) for tid, e in zip(token_ids, embeddings.tolist())
    ]

    with psycopg2.connect(
        dbname=POSTGRESQL_DB_NAME,
        user=POSTGRESQL_DB_USER,
        password=POSTGRESQL_DB_PASSWORD,
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.executemany(
                insert_query,
                insert_data
            )
            conn.commit()


def errors_insert(message: str) -> None:
    with psycopg2.connect(
        dbname=POSTGRESQL_DB_NAME,
        user=POSTGRESQL_DB_USER,
        password=POSTGRESQL_DB_PASSWORD,
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            curr_time = datetime.datetime.now(datetime.timezone.utc)
            cursor.execute(
                f"""INSERT INTO errors (message, timestamptz) 
                VALUES (%s, %s);""",
                (message, curr_time)
            )
            conn.commit()


def table_read(table_name: str) -> list[RealDictRow]:
    select_query: str = """SELECT * FROM %s ORDER BY id;"""
    with psycopg2.connect(
        dbname=POSTGRESQL_DB_NAME,
        user=POSTGRESQL_DB_USER,
        password=POSTGRESQL_DB_PASSWORD,
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(select_query, (table_name,))
            instances = cursor.fetchall()
    return instances


async def db_get_conn_async():
    return await asyncpg.connect(
        database=POSTGRESQL_DB_NAME,
        user=POSTGRESQL_DB_USER,
        password=POSTGRESQL_DB_PASSWORD,
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT
    )


def tokens_chunk_generator(model_name, chunk_size: int = 4096) -> Iterator[list[RealDictRow]]:
    select_query: str = """
        SELECT cmt.id, cmt.token
          FROM tokens AS cmt
          LEFT JOIN (SELECT * FROM embeddings WHERE model_name = %s) AS e
            ON cmt.id = e.token_id
         WHERE token_id IS NULL;
    """
    with psycopg2.connect(
            dbname=config.POSTGRESQL_DB_NAME,
            user=config.POSTGRESQL_DB_USER,
            password=config.POSTGRESQL_DB_PASSWORD,
            host=config.POSTGRESQL_DB_HOST,
            port=config.POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(select_query, (model_name,))
            while True:
                instances = cursor.fetchmany(chunk_size)
                if not instances:
                    break
                yield instances
