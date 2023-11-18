import os
import qdrant_client
import sanic
import json
from labrador import config
from labrador.util import utils, log
from labrador.dense.searcher import DenseSearcher

from labrador.config import QDRANT_HOST, QDRANT_GRPC_PORT

logger = log.configure_logger(__file__)
log = log.log(logger)

# app
app = sanic.Sanic("DenseSearcherAPI")
os.environ["SANIC_REQUEST_TIMEOUT"] = "30"
os.environ["SANIC_RESPONSE_TIMEOUT"] = "30"


@app.before_server_start
async def init_resources(app):
    app.ctx.model_name = list(config.MODELS.keys())[0]
    app.ctx.token_type = "sentence_with_keywords"
    app.ctx.collection_name = utils.collection_name(app.ctx.model_name, app.ctx.token_type)
    app.ctx.index_client = qdrant_client.QdrantClient(QDRANT_HOST, port=QDRANT_GRPC_PORT)
    app.ctx.neural_searcher = DenseSearcher(
        client=app.ctx.index_client,
        model_name=app.ctx.model_name,
        collection_name=app.ctx.collection_name,
        token_type=app.ctx.token_type,
    )


@app.route("/")
async def handle_neural_search(request: sanic.Request):
    response = await request.respond(content_type="application/json", status=200)
    try:
        logger.info(f"Handling request: {request.args}")
        query = request.args.get("query", "").strip()
        logger.info(f"query:{query}")
        hits = await app.ctx.neural_searcher.search_async(query)
        return await response.send(json.dumps({"hits": hits}))
    except Exception as e:
        logger.exception(f"Failed to handle request: {e}")
        return await response.send(json.dumps({"hits": []}))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8444, workers=1, auto_reload=True)
