import json
from typing import Optional, Iterable, Union

import os
import bs4
import asyncio
import random

import httpcore
import httpx
import qdrant_client
import sanic
from sanic.response import HTTPResponse
from sanic.request import Request
from sanic import response
from sanic_limiter import Limiter
# import sanic_jinja2 as jinja2

import config
from util import utils
from search.searcher import SearchResultObject
from search.ns import NeuralSearcher, LocalNeuralSearcher
from search.repository_searcher import RepositorySearcher

from config import APP_DIR, QDRANT_HOST, QDRANT_GRPC_PORT
from util import database, log

STATIC_DIR: str = os.path.join(APP_DIR, "static")
TEMPLATE_DIR: str = os.path.join(APP_DIR, "templates")

# logging
logger = log.configure_logger(__file__)
log = log.log(logger)

# app
app = sanic.Sanic("Labrador")
app.static("/static", STATIC_DIR)
limiter = Limiter(app)
# jinja = jinja2.SanicJinja2(app, loader=jinja2.FileSystemLoader(TEMPLATE_DIR))


@app.before_server_start
async def init_resources(app, loop):
    # config
    app.ctx.adb = database.AsyncDatabase()
    app.ctx.adb.conn_pool_init()
    app.ctx.model_name = list(config.MODELS.keys())[0]
    app.ctx.token_type = "sentence_with_keywords"

    app.ctx.collection_name = utils.collection_name(app.ctx.model_name, app.ctx.token_type)
    app.ctx._index_client = qdrant_client.QdrantClient(QDRANT_HOST, port=QDRANT_GRPC_PORT)
    app.ctx.neural_searcher = NeuralSearcher(
        client=app.ctx._index_client,
        model_name=app.ctx.model_name, collection_name=app.ctx.collection_name,
        token_type=app.ctx.token_type,
    )
    app.ctx.repository_searcher = RepositorySearcher(database=app.ctx.adb)
    app.ctx.keyword_searcher = None
    # app.ctx.keyword_searcher = KeywordSearcher()

    app.ctx.client_ip_table = {}

    logger.info("Resources initialized")


@app.listener('before_server_start')
async def start_periodic_tasks(app, loop):
    loop.create_task(periodic_tasks(app))


@app.get("/")
async def index(request: Request) -> HTTPResponse:
    return await response.file(os.path.join(STATIC_DIR, "index.html"))


@app.get("/search")
@limiter.limit("5 per minute")
async def search(request: Request) -> HTTPResponse:
    if disallow_ip(request.ip):
        return sanic.response.json({"success": False, "error": "Too many requests from this IP. Chill..."}, status=429)
    update_client_ip_table(app, request.ip)

    query = request.args.get("query", "").strip()
    if query == "":
        return sanic.response.json({"success": False, "error": "No query provided"}, status=400)

    response = await request.respond(content_type="application/json", status=200)
    query_id = 0
    sent_hits_ids: set[int] = set()
    try:
        gather_results = await asyncio.gather(
            *(app.ctx.adb.queries_write(query),
              app.ctx.neural_searcher.search_async(query))
        )
        query_id, ns_hits = gather_results
        structured_ns_hits = structure_hits(ns_hits, sent_hits_ids)

        await response.send(json.dumps({"success": True, "queryId": query_id, "hits": structured_ns_hits, "done": False}) + "\n")

        response = await request.respond(content_type="application/json", status=200)
        rs_hits = await app.ctx.repository_searcher.search_async(query)

        structured_rs_hits = structure_hits(rs_hits, sent_hits_ids)

        await response.send(json.dumps({"success": True, "queryId": query_id, "hits": structured_rs_hits, "done": False}) + "\n")

    except (TimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout):
        return sanic.response.json({"success": False, "error": "Search timed out"}, status=504)

    finally:
        response = await request.respond(content_type="application/json", status=200)
        await response.send(json.dumps({"success": True, "queryId": query_id, "hits": [], "done": True}) + "\n")


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
        logger.info("Running periodic tasks")
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
    app.run(host="0.0.0.0", port=8000,  debug=True, auto_reload=True)
    