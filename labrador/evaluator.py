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

import pytrec_eval
from psycopg2.extras import RealDictRow

from util import database as db

from labrador.util import log
logger = log.configure_logger(__file__)
log = log.log(logger)


class Evaluator:
    def __init__(self, searcher, top_k=10):
        self.searcher = searcher
        self.top_k = top_k
        self.qrels: list[RealDictRow] = db.qrels_read()

    @log
    def evaluate(self, queries, qrels=None):
        results = {
            'searcher': self.searcher.__class__.__name__,
        }
