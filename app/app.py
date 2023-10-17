from typing import Optional, Iterable

import os
import bs4
import asyncio
import random

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
from ingest.models import Webpage, Metadata
from search.searcher import NeuralSearcher, RepositorySearcher, SearchResultObject, LocalNeuralSearcher

from config import APP_DIR, QDRANT_HOST, QDRANT_PORT
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
    await database.async_conn_pool_init()
    database.conn_pool_init()
    app.ctx.model_name = list(config.MODELS.keys())[0]
    app.ctx.token_type = "sentence_with_keywords"

    app.ctx.collection_name = utils.collection_name(app.ctx.model_name, app.ctx.token_type)
    app.ctx.client = qdrant_client.QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    app.ctx.neural_searcher = NeuralSearcher(
        client=app.ctx.client,
        model_name=app.ctx.model_name, collection_name=app.ctx.collection_name,
        token_type=app.ctx.token_type
    )
    app.ctx.repository_searcher = RepositorySearcher()
    app.ctx.keyword_searcher = None
    # ks = KeywordSearcher()

    logger.info("Resources initialized")


@app.get("/")
async def index(request: Request) -> HTTPResponse:
    return await response.file(os.path.join(STATIC_DIR, "index.html"))


@app.get("/search/")
@limiter.limit("5 per minute")
async def search(request: Request) -> HTTPResponse:
    query = request.args.get("query", None).strip()
    if query is None:
        return sanic.response.json({"error": "No query provided"}, status=400)

    try:
        gather_results = await asyncio.gather(
            *(database.queries_write(query),
              app.ctx.repository_searcher.search_async(query),
              app.ctx.neural_searcher.search_async(query))
        )
    except httpx.ReadTimeout:
        return sanic.response.json({"error": "Search timed out"}, status=504)

    query_id = gather_results[0]
    logger.debug(f"Query ID: {query_id} hits")

    hits = [gather_results[1], gather_results[2]]
    logger.debug(f"Repository Searcher returned {len(hits[0])} hits")
    logger.debug(f"Neural Searcher returned {len(hits[1])} hits")

    structured_hits = structure_hits(hits)
    return sanic.response.json({"queryId": query_id, "hits": structured_hits})


@app.get("/annotate/")
@limiter.limit("40 per minute")
async def annotate(request: Request) -> HTTPResponse:
    query_id = request.args.get("query_id", None)
    if query_id is None:
        return sanic.response.json({"error": "No query_id provided"}, status=400)
    doc_id = request.args.get("doc_id", None)
    if doc_id is None:
        return sanic.response.json({"error": "No doc_id provided"}, status=400)
    rel = request.args.get("rel", None)
    if rel is None:
        return sanic.response.json({"error": "No rel provided"}, status=400)

    logger.info(f"Received annotation request for query_id: {query_id}")
    try:
        await database.qrels_write(query_id, doc_id, rel)
        return sanic.response.json({"success": True})
    except Exception as e:
        logger.exception(e)
        return sanic.response.json({"error": "Failed to write annotation"}, status=500)

app.static('/test/', os.path.join(TEMPLATE_DIR, "test.html"))


def structure_hits(list_of_hits: list[list[dict]]) -> list[dict]:
    hits = utils.flatten(list_of_hits)

    # TODO: This is a temporary solution for cleaning the title. The title should be cleaned in the database.
    #  The processor already implements this. The idea is for the processor to identify differences and update
    #  and for every other component to update based on change of data based on how it changed.
    for i, hit in enumerate(hits):
        hit['title'] = hit['title'].strip("\"")

    shuffled_hits = shuffle_hits(hits)
    return shuffled_hits


def shuffle_hits(hits: list[dict]) -> list[dict]:
    inserted = set()
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
    