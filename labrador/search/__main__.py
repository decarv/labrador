import time
import asyncio

import qdrant_client


import config
from search.ns import NeuralSearcher
from search.rs import RepositorySearcher
from util import database

query = "ataque cardíaco criança"


async def main_async():
    adb = database.AsyncDatabase()
    await adb.conn_pool_init()

    rs = RepositorySearcher()
    start = time.time()
    hit_list = await rs.search_async(query, 10)
    end = time.time()
    print("RS Async: ", end - start)

    ns = NeuralSearcher(
        client=qdrant_client.QdrantClient(url=config.QDRANT_HOST, port=config.QDRANT_GRPC_PORT),
        model_name=list(config.MODELS.keys())[0],
        token_type="sentence_with_keywords",
        language="pt",
    )
    start = time.time()
    hit_list = await ns.search_async(query, 10)
    end = time.time()
    print("NS Async: ", end - start)


def main():
    database.conn_pool_init()

    rs = RepositorySearcher()
    start = time.time()
    hit_list = rs.search(query)
    end = time.time()
    print("RS: ", end - start)

    ns = NeuralSearcher(
        client=qdrant_client.QdrantClient(url=config.QDRANT_HOST, port=config.QDRANT_GRPC_PORT),
        model_name=list(config.MODELS.keys())[0],
        token_type="sentence_with_keywords",
        language="pt",
    )
    start = time.time()
    hit_list = ns.search(query)
    end = time.time()
    print("NS: ", end - start)


if __name__ == "__main__":
    main()
    asyncio.run(main_async())
