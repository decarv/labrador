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
import asyncio
import datetime
from typing import Iterator, Optional, Union

import psycopg
import psycopg_pool
import asyncpg
import psycopg2
import torch
from psycopg2 import pool
from psycopg2.extras import RealDictRow, RealDictCursor, DictCursor

import config
from config import POSTGRESQL_DB_NAME, POSTGRESQL_DB_USER, POSTGRESQL_DB_PASSWORD, POSTGRESQL_DB_HOST, \
    POSTGRESQL_DB_PORT
from ingest.models import Webpage, Metadata

from util import log
import asyncio
from psycopg.rows import dict_row

logger = log.configure_logger(__file__)
log = log.log(logger)

conn_pool: Optional[pool.SimpleConnectionPool] = None
async_conn_pool: Optional[psycopg_pool.AsyncConnectionPool] = None
conninfo: str = (f"dbname={config.POSTGRESQL_DB_NAME} "
                 f"user={config.POSTGRESQL_DB_USER} "
                 f"password={config.POSTGRESQL_DB_PASSWORD} "
                 f"host={config.POSTGRESQL_DB_HOST} "
                 f"port={config.POSTGRESQL_DB_PORT}")


def conn_pool_init(minconn: int = 1, maxconn: int = 20):
    global conn_pool
    conn_pool = psycopg_pool.ConnectionPool(conninfo, min_size=minconn, max_size=maxconn,)


async def async_conn_pool_init():
    await _async_conn_pool_init()


async def _async_conn_pool_init(minconn: int = 1, maxconn: int = 20):
    global async_conn_pool
    async_conn_pool = psycopg_pool.AsyncConnectionPool(conninfo=conninfo, min_size=minconn, max_size=maxconn,)


def get_conn():
    global conn_pool
    if conn_pool is None:
        raise Exception("Connection pool is not initialized.")
    return conn_pool.getconn()


def put_conn(conn):
    global conn_pool
    if conn_pool is None:
        raise Exception("Connection pool is not initialized.")
    return conn_pool.putconn(conn)


async def get_conn_async():
    global async_conn_pool
    if async_conn_pool is None:
        raise Exception("Connection pool is not initialized.")
    return await async_conn_pool.getconn()


async def put_conn_async(conn):
    if conn is None:
        return
    global async_conn_pool
    if async_conn_pool is None:
        raise Exception("Connection pool is not initialized.")
    return await async_conn_pool.putconn(conn)


async def get_async_connection():
    return await psycopg.AsyncConnection.connect(conninfo)


def table_instance_exists(table: str, column: str, value: str) -> Optional[tuple]:
    conn = get_conn()
    cursor = conn.cursor()
    query = f"SELECT * FROM {table} WHERE {column} = %s;"
    cursor.execute(query, (value,))
    result = cursor.fetchone()
    put_conn(conn)
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
    put_conn(conn)


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
        put_conn(conn)


def webpages_empty() -> bool:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM webpages;")
    result = cursor.fetchone()
    put_conn(conn)
    return result[0] == 0


def webpages_insert(webpage: Webpage) -> None:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO webpages VALUES (%s, %s, %s, %s)",
        (webpage.url, webpage.is_metadata, webpage.is_document, 0)  # crawled = 0
    )
    conn.commit()
    put_conn(conn)


def documents_insert(metadata: Metadata) -> None:
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
    put_conn(conn)


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
            cursor.execute(f"SELECT * FROM data ORDER BY id;")
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
            cursor.execute(f"SELECT * FROM {table_name} order by id ;")
            while True:
                instances = cursor.fetchmany(chunk_size)
                if not instances:
                    break
                yield instances


def documents_chunk_generator(chunk_size: int = 4096) -> Iterator[list[RealDictRow]]:
    yield from table_chunk_generator("data", chunk_size)


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
    select_query: str = f"""SELECT * FROM {table_name} ORDER BY id;"""
    with psycopg2.connect(
        dbname=POSTGRESQL_DB_NAME,
        user=POSTGRESQL_DB_USER,
        password=POSTGRESQL_DB_PASSWORD,
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(select_query)
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


@log
async def embeddings_read_async(model_name: str, token_type: str, language: str) -> dict[int, tuple[list[float], str]]:
    global async_conn_pool
    if async_conn_pool is None:
        raise Exception("Connection pool is not initialized.")

    batch_size = 100000
    records: dict[int, tuple[list[float], str]] = {}

    with psycopg.Connection.connect(conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT t.data_id
                     FROM embeddings AS e
                     JOIN tokens AS t ON e.token_id = t.id
                    WHERE e.model_name = %s AND t.token_type = %s AND t.language = %s
                    ORDER BY t.data_id;""", (model_name, token_type, language,)
            )
            ids: list[tuple[int, ...]] = [r[0] for r in cur.fetchall()]
            interval_idx = list(range(0, len(ids), batch_size))
            interval_idx.append(len(ids)-1)
            intervals = [(ids[interval_idx[i]], ids[interval_idx[i+1]]) for i in range(len(interval_idx)-1)]
    await asyncio.gather(*[
        _embeddings_read(records,
                         (model_name, token_type, language, from_id, to_id,)) for from_id, to_id in intervals
    ])
    return records


async def _embeddings_read(records, _params: tuple[str, str, str, int, int]) -> None:
    language = _params[2]
    async with await psycopg.AsyncConnection.connect(conninfo) as aconn:
        async with aconn.cursor() as cur:
            await cur.execute(
                """SELECT t.data_id, e.embeddings
                    FROM embeddings AS e
                    JOIN tokens AS t ON e.token_id = t.id
                    JOIN data AS d ON t.data_id = d.id
                   WHERE e.model_name = %s AND t.token_type = %s AND t.language = %s AND t.data_id >= %s AND t.data_id < %s;""",
                _params)
            records = await cur.fetchall()
            for data_id, emb, tok, _ in records:
                records[data_id] = (emb, tok)


def embeddings_read(*params, batch_size: int = 128) -> Iterator[list[RealDictRow]]:
    select_query = """SELECT t.data_id, e.embeddings
            FROM embeddings AS e
            JOIN tokens AS t ON e.token_id = t.id
            JOIN data AS d ON t.data_id = d.id
           WHERE e.model_name = %s AND t.token_type = %s AND t.language = %s
           LIMIT %s;"""
    with psycopg2.connect(
            dbname=POSTGRESQL_DB_NAME,
            user=POSTGRESQL_DB_USER,
            password=POSTGRESQL_DB_PASSWORD,
            host=POSTGRESQL_DB_HOST,
            port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(select_query, (*params, batch_size,))
            while True:
                instances = cursor.fetchmany(batch_size)
                if not instances:
                    break
                yield instances


def data_read_in_ids(data_ids: list[int]):
    select_query = """SELECT d.id, d.title_pt, d.abstract_pt, d.keywords_pt
            FROM data as d
           WHERE d.id = ANY(%s);"""
    with psycopg2.connect(
            dbname=POSTGRESQL_DB_NAME,
            user=POSTGRESQL_DB_USER,
            password=POSTGRESQL_DB_PASSWORD,
            host=POSTGRESQL_DB_HOST,
            port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(select_query, (data_ids,))
            return cursor.fetchall()


def documents_query_url_path_suffix(urls: list[str]):
    if conn_pool is None:
        raise Exception("Connection pool is not initialized.")
    select_query = """SELECT d.id as doc_id, d.title_pt as title, d.abstract_pt as abstract, d.keywords_pt as keywords,
                                d.author, d.url
                    FROM data as d 
                    WHERE d.url_path_suffix = ANY(%s);"""
    with conn_pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(select_query, (urls,))
            records = cursor.fetchall()
            return records



async def documents_query_url_path_suffix_async(urls: list[str]) -> list[dict]:
    select_query = """SELECT d.id as doc_id, d.title_pt as title, d.abstract_pt as abstract, d.keywords_pt as keywords,
                                d.author, d.url
                    FROM data as d 
                    WHERE d.url_path_suffix = ANY(%s);"""
    aconn = await get_conn_async()
    acur = aconn.cursor(row_factory=dict_row)
    await acur.execute(select_query, (urls,))
    try:
        rows = await acur.fetchall()
        return rows
    except Exception as e:
        raise e
    finally:
        await acur.close()
        await put_conn_async(aconn)


def qrels_read() -> list[RealDictRow]:
    select_query = """SELECT * FROM qrels;"""
    with psycopg2.connect(
            dbname=POSTGRESQL_DB_NAME,
            user=POSTGRESQL_DB_USER,
            password=POSTGRESQL_DB_PASSWORD,
            host=POSTGRESQL_DB_HOST,
            port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(select_query)
            return cursor.fetchall()


async def qrels_write(query_id: Union[int, str], doc_id: Union[int, str], relevance: Union[int, str]) -> None:
    insert_query = """INSERT INTO qrels (query_id, data_id, relevance) VALUES (%s, %s, %s);"""
    aconn = await get_conn_async()
    async with aconn.cursor() as acur:
        try:
            await acur.execute(insert_query, (query_id, doc_id, relevance,))
            await aconn.commit()
        except Exception as e:
            raise e
        finally:
            await put_conn_async(aconn)


async def queries_write(query):
    insert_query = """INSERT INTO queries (query) VALUES (%s) ON CONFLICT DO NOTHING RETURNING id;"""
    aconn = await get_conn_async()
    async with aconn.cursor() as acur:
        try:
            await acur.execute("SELECT id FROM queries WHERE query = %s;", (query,))
            id_of_inserted_row = await acur.fetchone()
            if id_of_inserted_row is not None:
                id_of_inserted_row = id_of_inserted_row[0]
            else:
                await acur.execute(insert_query, (query,))
                id_of_inserted_row = await acur.fetchone()
                id_of_inserted_row = id_of_inserted_row[0]
                await aconn.commit()
        except Exception as e:
            raise e
        finally:
            await put_conn_async(aconn)
    return id_of_inserted_row

if __name__ == "__main__":
    print(documents_query_url_path_suffix(
        ["disponiveis/18/18148/tde-27082013-105058", "disponiveis/5/5139/tde-06052009-162539"]
    ))
