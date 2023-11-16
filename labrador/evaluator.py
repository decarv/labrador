"""
qrels: (q, d, r)
    q: query id
    d: document id
    r: relevance score
    (q, d) is unique
    r is in {0, 1, 2, 3, 4}
    0: not relevant
    1: poorly relevant
    2: fairly relevant
    3: highly relevant
    4: perfectly relevant
"""

import math

from util.log import configure_logger
from models.searcher import Searcher
import time

import qdrant_client


import config
from dense.searcher import DenseSearcher
from repository_searcher import RepositorySearcher
from util.database import Database, AsyncDatabase

logger = configure_logger(__file__)


class Evaluator:
    def __init__(self, database: Database):
        self._database: Database = database
        self.queries_records: list[dict] = self._database.select("""SELECT id, query FROM queries""")
        self.queries_ids: list[int] = [r['id'] for r in self.queries_records]

    def evaluate(self, searcher, top_k):
        precisions = []
        ndcgs = []
        for record in self.queries_records:
            try:
                hits_records = searcher.search(record['query'], top_k)
                qrels_records = self._database.select(
                    """SELECT doc_id, relevance FROM qrels WHERE query_id = %s ORDER BY relevance""",
                    (record['query_id'],)
                )
            except Searcher.SearcherError as e:
                logger.error(f"Searcher : {type(e)} : {e}")
                self._database.insert_error(f"Error inserting evaluation into database: {e}")
                continue

            hits = {r['doc_id']: r for r in hits_records}
            qrels = {r['doc_id']: r['relevance'] for r in qrels_records}
            ideal_hits = dict(sorted(qrels.items(), key=lambda x: x[1], reverse=True))

            precision, ndcg = self.measures(hits, ideal_hits)

            try:
                self._database.insert(
                    """
                    INSERT INTO evaluations (query_id, searcher, model_name, token_type, top_k, precision, ndcg, language)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (record['query_id'], searcher.__class__.__name__, searcher.model_name, searcher.token_type, top_k,
                     precision, ndcg, searcher.language,)
                )
            except Database.DatabaseError as e:
                logger.error(f"Error inserting evaluation into database: {e}")
                self._database.insert_error(f"Error inserting evaluation into database: {e}")

            precisions.append(precision)
            ndcgs.append(ndcg)

        print(f"Average Precision: {sum(precisions) / len(precisions)} | Average nDCG: {sum(ndcgs) / len(ndcgs)}")

    def measures(self, hits: dict[int, dict], qrels: dict[int, int], ideal_hits: dict[int, int]):
        dcg = 0
        precision = 0
        for i, hit in enumerate(hits.values()):
            if hit['doc_id'] in qrels:
                relevance = qrels[hit['doc_id']]
                if relevance > 3:
                    precision += 1
            else:
                relevance = 0
            dcg += (2 ** relevance - 1) / math.log2(i + 2)

        idcg = 0
        for i, hit in enumerate(ideal_hits):
            idcg += (2 ** ideal_hits[hit] - 1) / math.log2(i + 2)

        return precision / len(hits), dcg / idcg



query = "ataque cardíaco criança"


async def main_async():
    adb = AsyncDatabase()
    await adb.conn_pool_init()

    rs = RepositorySearcher()
    start = time.time()
    hit_list = await rs.search_async(query, 10)
    end = time.time()
    print("RS Async: ", end - start)

    ns = DenseSearcher(
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
    db = Database()
    #
    # rs = RepositorySearcher()
    # start = time.time()
    # hit_list = rs.search(query)
    # end = time.time()
    # print("RS: ", end - start)

    ns = DenseSearcher(
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
    # asyncio.run(main_async())

