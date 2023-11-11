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
import time
import datetime
from typing import Iterator, Optional, Union, Iterable

import psycopg
from psycopg.rows import dict_row
import psycopg_pool
import asyncpg
import psycopg2
import torch
from psycopg2 import pool
from psycopg2.extras import RealDictRow, RealDictCursor, DictCursor

import config
from config import POSTGRESQL_DB_NAME, POSTGRESQL_DB_USER, POSTGRESQL_DB_PASSWORD, POSTGRESQL_DB_HOST, \
    POSTGRESQL_DB_PORT
from ingest.models import RawData

from util.log import log, configure_logger
import asyncio

configure_logger(__file__)


class Database:
    """
    TODO: expand error handling
    """
    class DatabaseError(Exception):
        pass

    def __init__(self):
        self._conninfo: str = config.POSTGRESQL_URL
        self._conn_pool: Optional[psycopg_pool.ConnectionPool] = None
        self.conn_pool_init()

    def conn_pool_init(self, minconn: int = 1, maxconn: int = 30):
        self._conn_pool = psycopg_pool.ConnectionPool(conninfo=self._conninfo, min_size=minconn, max_size=maxconn,)

    def getconn(self):
        if self._conn_pool is None:
            raise Database.DatabaseError("Connection pool is not initialized.")
        return self._conn_pool.getconn()

    def putconn(self, conn):
        if self._conn_pool is None:
            raise Database.DatabaseError("Connection pool is not initialized.")
        return self._conn_pool.putconn(conn)

    def batch_generator(self, query: str, var_args: Optional[tuple] = None, batch_size: int = 32) -> Iterator[list[dict]]:
        conn = self.getconn()
        with conn.cursor(row_factory=dict_row) as cur:
            if var_args is None:
                cur.execute(query)
            else:
                cur.execute(query, var_args)
            while True:
                instances = cur.fetchmany(batch_size)
                if not instances:
                    break
                yield instances
        self.putconn(conn)

    def insert(self, query: str, params: tuple):
        conn = self.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                conn.commit()
        except Exception as e:
            raise Database.DatabaseError(f"Error inserting many rows: {e}")
        self.putconn(conn)

    def insert_many(self, query: str, params: list[tuple] = None):
        conn = self.getconn()
        try:
            with conn.cursor() as cur:
                cur.executemany(query, params)
                conn.commit()
        except Exception as e:
            raise Database.DatabaseError(f"Error inserting many rows: {e}")
        self.putconn(conn)

    def insert_error(self, message: str) -> None:
        conn = self.getconn()
        try:
            with conn.cursor(row_factory=dict_row) as cursor:
                curr_time = datetime.datetime.now(datetime.timezone.utc)
                cursor.execute(
                    f"""INSERT INTO errors (message, timestamptz) 
                    VALUES (%s, %s);""",
                    (message, curr_time)
                )
                conn.commit()
        except Exception as e:
            raise Database.DatabaseError(f"{type(e)} : {e}")
        finally:
            self.putconn(conn)

    def select(self, query, var_args=None):
        conn = self.getconn()
        with conn.cursor(row_factory=dict_row) as cur:
            if var_args is None:
                cur.execute(query)
            else:
                cur.execute(query, var_args)
            results = cur.fetchall()
        self.putconn(conn)
        return results

class AsyncDatabase:
    class AsyncDatabaseError(Exception):
        pass

    def __init__(self):
        self._conninfo: str = (
            f"dbname={config.POSTGRESQL_DB_NAME} "
            f"user={config.POSTGRESQL_DB_USER} "
            f"password={config.POSTGRESQL_DB_PASSWORD} "
            f"host={config.POSTGRESQL_DB_HOST} "
            f"port={config.POSTGRESQL_DB_PORT}"
        )
        self._async_conn_pool: Optional[psycopg_pool.AsyncConnectionPool] = None
        self.conn_pool_init()

    def conn_pool_init(self, minconn: int = 1, maxconn: int = 30):
        self._async_conn_pool = psycopg_pool.AsyncConnectionPool(conninfo=self._conninfo, min_size=minconn, max_size=maxconn,)

    async def getconn(self, timeout: int = 100):
        if self._async_conn_pool is None:
            raise AsyncDatabase.AsyncDatabaseError("Connection pool is not initialized.")
        return await self._async_conn_pool.getconn()

    async def putconn(self, conn):
        if self._async_conn_pool is None:
            raise AsyncDatabase.AsyncDatabaseError("Connection pool is not initialized.")
        return await self._async_conn_pool.putconn(conn)

    async def batch_generator(self, query: str, batch_size: int = 1024) -> Iterator[list]:
        conn = await self.getconn()
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query)
            while True:
                instances = await cur.fetchmany(batch_size)
                if not instances:
                    break
                yield instances
        await self.putconn(conn)

    async def select(self, query, var_args=None):
        conn = await self.getconn()
        async with conn.cursor(row_factory=dict_row) as cur:
            if var_args is None:
                await cur.execute(query)
            else:
                await cur.execute(query, var_args)
            results = await cur.fetchall()
        await self.putconn(conn)
        return results

    async def insert(self, query, var_args=None):
        conn = await self.getconn()
        async with conn.cursor() as cur:
            if var_args is None:
                await cur.execute(query)
            else:
                await cur.execute(query, var_args)
            await cur.commit()
        await self.putconn(conn)

    async def queries_write(self, query):
        select_query = """SELECT id FROM queries WHERE query = %s;"""
        insert_query = """INSERT INTO queries (query) VALUES (%s) RETURNING id;"""
        aconn = await self.getconn()
        async with aconn.cursor() as acur:
            try:
                await acur.execute(select_query, (query,))
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
                await self.putconn(aconn)
        return id_of_inserted_row

    async def qrels_write(self, query_id: Union[int, str], doc_id: Union[int, str], relevance: Union[int, str]) -> None:
        insert_query = """INSERT INTO qrels (query_id, doc_id, relevance) VALUES (%s, %s, %s);"""
        aconn = await self.getconn()
        async with aconn.cursor() as acur:
            try:
                await acur.execute(insert_query, (query_id, doc_id, relevance,))
                await aconn.commit()
            except Exception as e:
                raise e
            finally:
                await self.putconn(aconn)


def conn_pool_init(minconn: int = 1, maxconn: int = 20):
    global conn_pool
    conn_pool = psycopg_pool.ConnectionPool(conninfo, min_size=minconn, max_size=maxconn,)

def get_conn(timeout=180):
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


def raw_data_contains(url: str) -> bool:
    return table_instance_exists("raw_data", "url", url) is not None


def webpages_crawled_update(webpage) -> None:
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


def webpages_insert(webpage) -> None:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO webpages VALUES (%s, %s, %s, %s)",
        (webpage.url, webpage.is_metadata, webpage.is_document, 0)  # crawled = 0
    )
    conn.commit()
    put_conn(conn)


def raw_data_insert(rd: RawData) -> None:
    conn = connection()
    with conn.cursor() as cur:
        cur.execute("""
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
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            rd.url,
            rd.doi,
            rd.type,
            rd.author,
            rd.institute,
            rd.knowledge_area,
            rd.committee,
            rd.title_pt,
            rd.title_en,
            rd.keywords_pt,
            rd.keywords_en,
            rd.abstract_pt,
            rd.abstract_en,
            rd.publish_date
        ))
    conn.commit()
    conn.close()


def raw_data_batch_insert(rd_list: Iterable[RawData]) -> None:
    conn = connection()
    with conn.cursor() as cur:
        query = """INSERT INTO raw_data (
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
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
        cur.executemany(query, [tuple(rd) for rd in rd_list])
    conn.commit()
    conn.close()


async def raw_data_batch_insert_async(rd_list: Iterable[RawData]) -> None:
    query = """INSERT INTO raw_data (
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
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)"""
    conn = await asyncpg.connect(
        host=POSTGRESQL_DB_HOST,
        port=POSTGRESQL_DB_PORT,
        user=POSTGRESQL_DB_USER,
        password=POSTGRESQL_DB_PASSWORD,
        database=POSTGRESQL_DB_NAME,
    )
    await conn.executemany(query, [tuple(rd) for rd in rd_list])
    await conn.close()


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
    yield from table_chunk_generator("clean_data", chunk_size)


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


def tokens_chunk_generator(token_type, chunk_size: int = 4096) -> Iterator[list[RealDictRow]]:
    select_query: str = """
        SELECT t.id, t.token, t.token_type, t.language, t.data_id, iq.collection_name,
        d.title_pt as title, d.abstract_pt as abstract, d.keywords_pt as keywords,
        d.author, d.url
          FROM tokens AS t
          JOIN clean_data AS d ON t.data_id = d.id
          LEFT OUTER JOIN indexed_in_qdrant AS iq ON t.id = iq.id
         WHERE token_type = %s
         AND iq.collection_name IS NULL;
    """
    with psycopg2.connect(
            dbname=config.POSTGRESQL_DB_NAME,
            user=config.POSTGRESQL_DB_USER,
            password=config.POSTGRESQL_DB_PASSWORD,
            host=config.POSTGRESQL_DB_HOST,
            port=config.POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(select_query, (token_type,))
            while True:
                instances = cursor.fetchmany(chunk_size)
                if not instances:
                    break
                yield instances


async def insert_clean_data(self, record: tuple):
    insert_query = """
               INSERT INTO clean_data (
               url_path_suffix,
               doi,
               type,
               author,
               institute,
               knowledge_area,
               committee,
               title, 
               keywords,
               abstract, 
               language,
               publish_date,
               raw_data_id,
               last_update
               ) 
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, to_date($14, 'YYYY-MM-DD'), $15, $16)
               ON CONFLICT (raw_data_id)
               DO UPDATE SET 
               keywords_pt = EXCLUDED.keywords_pt;
               """
    aconn = await get_conn_async()
    async with aconn.cursor() as acur:
        try:
            await acur.execute(insert_query, record)
            await aconn.commit()
        except Exception as e:
            raise e
        finally:
            await put_conn_async(aconn)


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


async def read_raw_data() -> None:
    select_query = """SELECT * FROM raw_data WHERE cleaned = false ORDER BY id;"""
    aconn = await get_conn_async()
    async with aconn.cursor() as acur:
        data_records = await acur.execute(select_query)
        data_records = await acur.fetchall()
        return data_records


def raw_data_get_urls(batch_size: int = 4096) -> list[str]:
    select_query = """SELECT url FROM raw_data ORDER BY id;"""
    conn = connection()
    with conn.cursor() as cur:
        cur.execute(select_query)
        urls = cur.fetchall()
    conn.close()
    return [url[0] for url in urls]


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



def raw_data_batch_generator(batch_size: int = 64) -> Iterator[list[RealDictRow]]:
    select_query = """SELECT * FROM raw_data WHERE cleaned = false ORDER BY id;"""
    with psycopg2.connect(
            dbname=POSTGRESQL_DB_NAME,
            user=POSTGRESQL_DB_USER,
            password=POSTGRESQL_DB_PASSWORD,
            host=POSTGRESQL_DB_HOST,
            port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(select_query, (batch_size,))
            while True:
                instances = cursor.fetchmany(batch_size)
                if not instances:
                    break
                yield instances


def clean_data_batch_insert(records: list[dict]):
    columns = ", ".join(records[0].keys())
    vars_list = [list(record.values()) for record in records]
    insert_query = f"""INSERT INTO clean_data ({columns}) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, to_date(%s, 'YYYY-MM-DD'), %s, %s)
                        ON CONFLICT (raw_data_id)
                        DO NOTHING;
                    """
    with psycopg2.connect(
            dbname=POSTGRESQL_DB_NAME,
            user=POSTGRESQL_DB_USER,
            password=POSTGRESQL_DB_PASSWORD,
            host=POSTGRESQL_DB_HOST,
            port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor() as cur:
            cur.executemany(insert_query, vars_list)
            cur.executemany("UPDATE raw_data SET cleaned = true WHERE id = %s;",
                            [(record['raw_data_id'],) for record in records])
            conn.commit()


def id_exists_in_qdrant(id: int, collection_name: str) -> bool:
    select_query = """SELECT * FROM indexed_in_qdrant WHERE id = %s AND collection_name = %s;"""
    with psycopg2.connect(
            dbname=POSTGRESQL_DB_NAME,
            user=POSTGRESQL_DB_USER,
            password=POSTGRESQL_DB_PASSWORD,
            host=POSTGRESQL_DB_HOST,
            port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(select_query, (id, collection_name,))
            return len(cursor.fetchall()) > 0

def insert_batch_in_qdrant(records: list[tuple[int, str]]) -> None:
    insert_query = """INSERT INTO indexed_in_qdrant (id, collection_name) VALUES (%s, %s);"""
    with psycopg2.connect(
            dbname=POSTGRESQL_DB_NAME,
            user=POSTGRESQL_DB_USER,
            password=POSTGRESQL_DB_PASSWORD,
            host=POSTGRESQL_DB_HOST,
            port=POSTGRESQL_DB_PORT
    ) as conn:
        with conn.cursor() as cursor:
            cursor.executemany(insert_query, records)
            conn.commit()


if __name__ == "__main__":
    pass