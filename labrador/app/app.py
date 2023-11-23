import json
from typing import Optional

import os
import asyncio
import random

import httpx
import pysolr
import sanic
from sanic.response import HTTPResponse
from sanic.request import Request
from sanic import response
from sanic_limiter import Limiter

from labrador import config
from labrador.util import database, log
from labrador.repo.searcher import RepositorySearcher
from labrador.sparse.searcher import SparseSearcher

from labrador.config import APP_DIR, CERTS_DIR

STATIC_DIR: str = os.path.join(APP_DIR, "static")
TEMPLATE_DIR: str = os.path.join(APP_DIR, "templates")

# logging
logger = log.configure_logger(__file__)
log = log.log(logger)

# app
app = sanic.Sanic("Labrador")
os.environ["SANIC_REQUEST_TIMEOUT"] = "30"
os.environ["SANIC_RESPONSE_TIMEOUT"] = "30"

app.static("/static", STATIC_DIR)
limiter = Limiter(app)


@app.before_server_start
async def init_resources(app, loop):
    # config
    app.ctx.adb = database.AsyncDatabase()
    app.ctx.adb.conn_pool_init()
    app.ctx.repository_searcher = RepositorySearcher(database=app.ctx.adb)
    app.ctx.sparse_retriever = SparseSearcher(client=pysolr.Solr(config.SOLR_URL))
    app.ctx.client_ip_table = {}
    app.ctx.shared_resources = {}
    logger.info("Resources initialized")


@app.listener('before_server_start')
async def start_periodic_tasks(app, loop):
    loop.create_task(periodic_tasks(app))


@app.get("/")
async def index(request: Request) -> HTTPResponse:
    return await response.file(os.path.join(STATIC_DIR, "index.html"))


# @app.get("/search")
# @limiter.limit("15 per minute")
# async def search(request: Request) -> HTTPResponse:
#     if disallow_ip(request.ip):
#         return sanic.response.json({"success": False, "error": "Too many requests from this IP. Chill..."}, status=429)
#     update_client_ip_table(app, request.ip)
#
#     query = request.args.get("query", "").strip()
#     if query == "":
#         return sanic.response.json({"success": False, "error": "No query provided"}, status=400)
#
#     response = await request.respond(content_type="application/json", status=200)
#     query_id = 0
#     sent_hits_ids: set[int] = set()
#     try:
#         gather_results = await asyncio.gather(
#             *(app.ctx.adb.queries_write(query),
#               app.ctx.neural_searcher.search_async(query))
#         )
#         query_id, hits = gather_results
#         structured_hits = structure_hits(hits, sent_hits_ids)
#
#         for hit in structured_hits:
#             hit['query_id'] = hit
#         await response.send(json.dumps({"success": True, "hits": structured_hits, "done": False}) + "\n")
#
#         response = await request.respond(content_type="application/json", status=200)
#         hits = await app.ctx.repository_searcher.search_async(query)
#
#         structured_rs_hits = structure_hits(hits, sent_hits_ids)
#
#         await response.send(json.dumps({"success": True, "queryId": query_id, "hits": structured_rs_hits, "done": False}) + "\n")
#
#     except (TimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
#         return sanic.response.json({"success": False, "error": f"Search timed out: {e}"}, status=504)
#
#     finally:
#         response = await request.respond(content_type="application/json", status=200)
#         await response.send(json.dumps({"success": True, "queryId": query_id, "hits": [], "done": True}) + "\n")


@app.get("/neural_search")
@limiter.limit("15 per minute")
async def neural_search(request: Request) -> HTTPResponse:
    if disallow_ip(request.ip):
        return sanic.response.json({"success": False, "error": "Too many requests from this IP. Chill..."}, status=429)
    update_client_ip_table(app, request.ip)
    query = request.args.get("query", "").strip()
    uid = request.args.get("uid", "").strip()
    if query == "":
        return sanic.response.json({"success": False, "error": "No query provided"}, status=400)
    if uid == "":
        return sanic.response.json({"success": False, "error": "No uid"}, status=400)
    if uid not in app.ctx.shared_resources:
        app.ctx.shared_resources[uid] = {}
        app.ctx.shared_resources[uid]['sent_hits'] = set()

    query_id = 0
    response = await request.respond(content_type="application/json", status=200)
    try:
        query_id, hits = await asyncio.gather(
            *(app.ctx.adb.queries_write(query),
              request_neural_search(query))
        )

        params = [(query_id, hit['doc_id'], 0) for hit in hits]
        await app.ctx.adb.insert_many(
            """INSERT INTO qrels (query_id, doc_id, relevance) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING""",
            params
        )

        structured_hits = structure_hits(hits, app.ctx.shared_resources[uid]['sent_hits'])
        for hit in structured_hits:
            hit['query_id'] = query_id

        await response.send(json.dumps({"success": True, "hits": structured_hits}) + "\n")

    except (TimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
        return response.json({"success": False, "error": f"Neural search timed out: {e}"}, status=504)


async def request_neural_search(query):
    async with httpx.AsyncClient() as client:
        response = await client.get("http://0.0.0.0:8444/", params={"query": query})
        response_text = await response.aread()
        response_json = json.loads(response_text)
        return response_json['hits']


@app.get("/keyword_search")
@limiter.limit("15 per minute")
async def keyword_search(request: Request) -> HTTPResponse:
    if disallow_ip(request.ip):
        return sanic.response.json({"success": False, "error": "Too many requests from this IP. Chill..."}, status=429)
    update_client_ip_table(app, request.ip)
    query = request.args.get("query", "").strip()
    uid = request.args.get("uid", "").strip()
    if query == "":
        return sanic.response.json({"success": False, "error": "No query provided"}, status=400)
    if uid == "":
        return sanic.response.json({"success": False, "error": "No uid"}, status=400)

    if uid not in app.ctx.shared_resources:
        app.ctx.shared_resources[uid] = {}
        app.ctx.shared_resources[uid]['sent_hits'] = set()

    query_id = 0
    response = await request.respond(content_type="application/json", status=200)
    try:
        query_id = await app.ctx.adb.queries_write(query)
        hits = app.ctx.sparse_retriever.search(query, top_k=10)

        params = [(query_id, hit['doc_id'], 0) for hit in hits]
        await app.ctx.adb.insert_many(
            """INSERT INTO qrels (query_id, doc_id, relevance) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING""",
            params
        )

        structured_hits = structure_hits(hits, app.ctx.shared_resources[uid]['sent_hits'])
        for hit in structured_hits:
            hit['query_id'] = query_id
        await response.send(json.dumps({"success": True, "hits": structured_hits}) + "\n")

    except (TimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout):
        return response.json({"success": False, "error": "Keyword search timed out"}, status=504)


@app.get("/repository_search")
@limiter.limit("15 per minute")
async def repository_search(request: Request) -> HTTPResponse:
    if disallow_ip(request.ip):
        return sanic.response.json({"success": False, "error": "Too many requests from this IP. Chill..."}, status=429)
    update_client_ip_table(app, request.ip)
    query = request.args.get("query", "").strip()
    uid = request.args.get("uid", "").strip()
    if query == "":
        return sanic.response.json({"success": False, "error": "No query provided"}, status=400)
    if uid == "":
        return sanic.response.json({"success": False, "error": "No uid"}, status=400)

    if uid not in app.ctx.shared_resources:
        app.ctx.shared_resources[uid] = {}
        app.ctx.shared_resources[uid]['sent_hits'] = set()

    query_id = 0
    response = await request.respond(content_type="application/json", status=200)
    try:
        query_id, hits = await asyncio.gather(
            *(app.ctx.adb.queries_write(query),
              app.ctx.repository_searcher.search_async(query))
        )

        params = [(query_id, hit['doc_id'], 0) for hit in hits]
        await app.ctx.adb.insert_many(
            """INSERT INTO qrels (query_id, doc_id, relevance) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING""",
            params
        )

        structured_hits = structure_hits(hits, app.ctx.shared_resources[uid]['sent_hits'])
        for hit in structured_hits:
            hit['query_id'] = query_id
        await response.send(json.dumps({"success": True, "hits": structured_hits}) + "\n")
    except (TimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout):
        return response.json({"success": False, "error": "Repository search timed out"}, status=504)


@app.get("/missing")
@limiter.limit("40 per minute")
async def return_missing(request: Request) -> HTTPResponse:
    if disallow_ip(request.ip):
        return sanic.response.json({"error": "Too many requests from this IP. Chill..."}, status=429)
    update_client_ip_table(app, request.ip)
    uid = request.args.get("uid", "").strip()
    if uid == "":
        return sanic.response.json({"success": False, "error": "No uid"}, status=400)

    response = await request.respond(content_type="application/json", status=200)
    try:
        misses: list[dict] = await app.ctx.adb.select(
            """
            SELECT q.id as query_id, d.id as doc_id, d.title_pt as title, d.abstract_pt as abstract, d.keywords_pt as keywords,
            d.author, q.query
              FROM queries AS q
              JOIN qrels AS qr ON q.id = qr.query_id
              JOIN documents AS d ON qr.doc_id = d.id
             WHERE qr.relevance = 0
             LIMIT 5;
            """
        )
        await response.send(json.dumps({"success": True, "hits": misses}))
    except (TimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout):
        return response.json({"success": False, "error": "Failed to retrieve misses"}, status=504)


@app.get("/annotate")
@limiter.limit("40 per minute")
async def annotate(request: Request) -> HTTPResponse:
    if disallow_ip(request.ip):
        return sanic.response.json({"error": "Too many requests from this IP. Chill..."}, status=429)
    update_client_ip_table(app, request.ip)

    query_id, doc_id, rel, rc = validate_input(request)
    if rc is not None:
        return rc

    logger.info(f"Received annotation request for query_id: {query_id}")
    try:
        await app.ctx.adb.qrels_write(query_id, doc_id, rel)
        return sanic.response.json({"success": True})
    except Exception as e:
        logger.exception(e)
        await app.ctx.adb.insert_error(e)
        return sanic.response.json({"error": "Failed to write annotation"}, status=500)


def update_client_ip_table(app, ip: str):
    if ip not in app.ctx.client_ip_table:
        app.ctx.client_ip_table[ip] = {
            'recent_requests': 0,
            'blacklist': False
        }
    app.ctx.client_ip_table[ip]['recent_requests'] += 1


async def periodic_tasks(app):
    while True:
        client_ip_table_cleanup(app)
        await asyncio.sleep(300)


def client_ip_table_cleanup(app):
    for ip in app.ctx.client_ip_table.keys():
        app.ctx.client_ip_table[ip]['recent_requests'] = 0
        app.ctx.client_ip_table[ip]['blacklist'] = False


def disallow_ip(ip: str) -> bool:
    if ip in app.ctx.client_ip_table:
        return app.ctx.client_ip_table[ip]['recent_requests'] > 100 or app.ctx.client_ip_table[ip]['blacklist']
    return False


def validate_input(request: Request) -> tuple[int, int, int, Optional[HTTPResponse]]:

    query_id = request.args.get("query_id", None).strip()
    doc_id = request.args.get("doc_id", None)
    rel = request.args.get("rel", None)

    if query_id is None or doc_id is None or rel is None:
        return 0, 0, 0, sanic.response.json({"error": "Malformed input"}, status=400)

    try:
        query_id = int(query_id)
        rel = int(rel)
        doc_id = int(doc_id)
        assert rel in [1, 2, 3, 4, 5]
    except ValueError:
        app.ctx.client_ip_table[request.ip]['blacklist'] = True
        return 0, 0, 0, sanic.response.json({"error": "Malformed input"}, status=400)
    except AssertionError:
        app.ctx.client_ip_table[request.ip]['blacklist'] = True
        return 0, 0, 0, sanic.response.json({"error": "Malformed input"}, status=400)

    return query_id, doc_id, rel, None


def structure_hits(hits: list[dict], sent_hits_ids: set[int]) -> list[dict]:
    # hits = utils.flatten(list_of_hits)

    # TODO: This is a temporary solution for cleaning the title. The title should be cleaned in the database.
    #  The processor already implements this. The idea is for the processor to identify differences and update
    #  and for every other component to update based on change of data based on how it changed.
    for i, hit in enumerate(hits):
        hit['title'] = hit['title'].strip("\"")
    shuffled_hits = shuffle_hits(hits, sent_hits_ids)
    return shuffled_hits

def shuffle_hits(hits: list[dict], inserted: set[int]) -> list[dict]:
    shuffled_hits = []
    for hit in hits:
        if hit['doc_id'] not in inserted:
            shuffled_hits.append(hit)
            inserted.add(hit['doc_id'])

    random.shuffle(shuffled_hits)
    return shuffled_hits

    # results: list[SearchResultObject] = []
    # results_set: set[str] = set()
    # for i in range(n):
    #     wr_i = wr_idxs[i]
    #     if website_results[wr_i]['url'] not in results_set:
    #         results.append([wr_i, website_results[wr_i]])
    #         results_set.add(website_results[wr_i]['url'])
    #     er_j = er_idxs[i]
    #     if engine_results[er_j]['url'] not in results_set:
    #         results.append([er_j, engine_results[er_j]])
    #         results_set.add(engine_results[er_j]['url'])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8443, ssl=CERTS_DIR, debug=True, auto_reload=True, workers=5)
