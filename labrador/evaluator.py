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
import os
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt

import qdrant_client
import pysolr

from labrador import config
from labrador.models.searcher import Searcher
from labrador.dense.tokenizer import Tokenizer
from labrador.dense.searcher import DenseSearcher
from labrador.sparse.searcher import SparseSearcher
from labrador.repo.searcher import RepositorySearcher
from labrador.util.database import Database
from labrador.util.log import configure_logger

logger = configure_logger(__file__)


class Evaluator:
    """
    A class for evaluating search query performance against a database of queries and relevance judgments (qrels).

    This class provides functionality to evaluate search queries using precision, nDCG, and timing measurements.
    It supports querying against a search interface, calculating evaluation metrics, and storing results in a database.

    Attributes:
        _database (Database): A database connection to query and store evaluation results.
        queries_records (list[dict]): Records of queries fetched from the database.
        queries_ids (list[int]): List of query IDs for evaluation.
    """
    def __init__(self, database: Database):
        """
        Initializes the Evaluator with a database connection.

        Args:
            database (Database): A database connection used for querying and storing evaluation data.
        """
        self._database: Database = database
        self.queries_records: list[dict] = self._database.select(
            """SELECT id as query_id, query 
                 FROM queries 
                ORDER BY query_id"""
        )
        self.queries_ids: list[int] = [r['query_id'] for r in self.queries_records]
        self.date_str: str = datetime.now().strftime('%Y-%m-%d')

    def evaluate(self, searcher, top_k):
        """
        Evaluates search queries using the specified searcher and metrics.

        Iterates through each query, performs the search, and calculates precision, nDCG, and timing metrics.
        Results are stored in the database, and average precision and nDCG are printed.

        Args:
            searcher: The search interface to perform queries.
            top_k (int): The number of top results to consider for evaluation.
        """
        logger.info(f"Evaluating {searcher.__class__.__name__} | {searcher.model_name} | {searcher.token_type} ")
        for query_record in self.queries_records:

            try:
                qrels_records = self._database.select(
                    """SELECT query_id, doc_id, relevance FROM qrels WHERE query_id = %s ORDER BY relevance""",
                    (query_record['query_id'],)
                )
            except Database.DatabaseError as e:
                logger.error(f"Error fetching qrels from database: {e}")
                self._database.insert_error(f"Error fetching qrels from database: {e}")
                continue

            qrels = {r['doc_id']: r['relevance'] for r in qrels_records}

            # Skip queries with no relevance scores
            if len(qrels) == 0:
                continue

            if self._is_evaluated(searcher, query_record['query_id'], top_k, len(qrels)):
                continue

            try:
                hits_records = searcher.search(query_record['query'], top_k)
            except Searcher.SearcherError as e:
                logger.error(f"Searcher : {type(e)} : {e}")
                self._database.insert_error(f"Error inserting evaluation into database: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected Error : {type(e)} : {e}")
                self._database.insert_error(f"Unexpected Error : {type(e)} : {e} : {searcher.__class__.__name__} "
                                            f": {searcher.model_name} {searcher.token_type} {searcher.language}")
                continue

            if len(hits_records) == 0:
                return

            meantime, stdtime = self._time_measurements(searcher, query_record['query'], top_k)

            hits = {r['doc_id']: r for r in hits_records}

            precision, ndcg, misses = self._relevance_measurements(hits, qrels)

            try:
                self._database.insert(
                    """INSERT INTO evaluations (
                                query_id, searcher, model_name, token_type, language, top_k, qrels_count, date,
                                precision, ndcg, misses, mean_time, stddev_time
                             )
                             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""",
                    (
                        query_record['query_id'], searcher.__class__.__name__, searcher.model_name,
                        searcher.token_type, searcher.language, top_k, len(qrels), self.date_str,
                        precision, ndcg, misses, meantime, stdtime,)
                )
            except Database.DatabaseError as e:
                logger.error(f"Error inserting evaluation into database: {e}")
                self._database.insert_error(f"Error inserting evaluation into database: {e}")

    def create_reports(self):
        evaluations = self._database.select(
            """SELECT searcher, model_name, token_type, AVG(ndcg) as avg_ndcg, AVG(mean_time) avg_mean_time,
                AVG(stddev_time) avg_stddev_time, AVG(precision) as avg_precision, AVG(misses) as avg_misses
                     FROM evaluations
                    GROUP BY searcher, model_name, token_type, date
                    ORDER BY date DESC;"""
        )
        latest_eval = evaluations[0]

        df = pd.DataFrame(latest_eval)

        self._create_table_report(df)
        self._plot_performance_heatmap(df)
        self._plot_model_comparison(df)
        self._plot_metric_distribution(df)
        self._plot_time_efficiency(df)
        self._plot_misses_vs_performance(df)

    def _create_table_report(self, df: pd.DataFrame) -> None:
        report_path = os.path.join(config.REPORTS_DIR, f"{self.date_str}_report.csv")
        df.to_csv(report_path)

    def _plot_performance_heatmap(self, heatmap_data):
        heatmap_data_pivot = heatmap_data.pivot("model_name", "searcher", "avg_ndcg")

        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data_pivot, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("Average nDCG by Searcher and Model")
        plt.ylabel("Model Name")
        plt.xlabel("Searcher")
        plt.savefig(os.path.join(config.REPORTS_DIR, f"{self.date_str}_heatmap_ndcg.png"))
        plt.close()

    def _plot_model_comparison(self, data: pd.DataFrame):

        plt.figure(figsize=(12, 6))
        sns.barplot(x="model_name", y="avg_precision", hue="searcher", data=data)
        plt.title("Average Precision by Model and Searcher")
        plt.xlabel("Model Name")
        plt.ylabel("Average Precision")
        plt.legend(title="Searcher")
        plt.savefig(os.path.join(config.REPORTS_DIR, f"{self.date_str}_model_comparison_precision.png"))
        plt.close()

    def _plot_metric_distribution(self, data: pd.DataFrame) -> None:
        plt.figure(figsize=(10, 6))
        sns.histplot(data, x="avg_ndcg", bins=30, kde=True)
        plt.title("Distribution of Average nDCG")
        plt.xlabel("Average nDCG")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(config.REPORTS_DIR, f"{self.date_str}_ndcg_distribution.png"))
        plt.close()

    def _plot_time_efficiency(self, data: pd.DataFrame):

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="avg_mean_time", y="avg_ndcg", hue="searcher", data=data)
        plt.title("Search Time Efficiency vs. nDCG")
        plt.xlabel("Average Mean Time")
        plt.ylabel("Average nDCG")
        plt.legend(title="Searcher")
        plt.savefig(os.path.join(config.REPORTS_DIR, f"{self.date_str}_time_efficiency_and_ndcg.png"))
        plt.close()

    def _plot_misses_vs_performance(self, data):

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="avg_misses", y="avg_precision", hue="searcher", data=data)
        plt.title("Misses vs. Average Precision")
        plt.xlabel("Average Misses")
        plt.ylabel("Average Precision")
        plt.legend(title="Searcher")
        plt.savefig(os.path.join(config.REPORTS_DIR, f"{self.date_str}_misses_vs_precicion.png"))
        plt.close()

    @staticmethod
    def _relevance_measurements(hits: dict[int, dict], qrels: dict[int, int]):
        ideal_hits = dict(sorted(qrels.items(), key=lambda x: x[1], reverse=True))

        dcg: float = 0
        precision: float = 0
        misses: int = 0
        for i, hit in enumerate(hits.values()):
            if hit['doc_id'] in qrels:
                relevance = qrels[hit['doc_id']]
                if relevance > 3:
                    precision += 1
            else:
                misses += 1
                relevance = 0
            dcg += (2 ** relevance - 1) / math.log2(i + 2)

        idcg = 0
        for i, hit in enumerate(ideal_hits):
            idcg += (2 ** ideal_hits[hit] - 1) / math.log2(i + 2)

        return precision / len(hits), dcg / idcg, misses

    @staticmethod
    def _time_measurements(searcher, query, top_k) -> tuple[float, float]:
        # TODO: Verificar se o tempo de execução é amenizado para runs a partir da primeira
        #   por conta do cache da GPU
        num_runs = 50
        times = []
        for i in range(num_runs):
            start = time.time()
            searcher.search(query, top_k)
            end = time.time()
            times.append(end - start)
        return np.mean(times), np.std(times)

    def _is_evaluated(self, searcher, query_id, top_k, qrels_count) -> bool:
        query_result = self._database.select(
            """SELECT query_id, searcher, model_name, token_type, language, top_k, qrels_count
                       FROM evaluations 
                      WHERE query_id = %s
                        AND searcher = %s
                        AND (model_name = %s OR model_name IS NULL)
                        AND (token_type = %s OR token_type IS NULL)
                        AND (language = %s OR language IS NULL)
                        AND top_k = %s
                        AND qrels_count = %s
                        ORDER BY date DESC;""",
            (query_id, searcher.__class__.__name__, searcher.model_name,
             searcher.token_type, searcher.language, top_k, qrels_count)
        )
        if len(query_result) == 0:
            return False
        existing_evaluation_qrels_count = query_result[0]['qrels_count']

        updated_query_result = self._database.select(
            """SELECT count(id) as count FROM qrels WHERE query_id = %s;""",
            (query_id,)
        )
        if len(updated_query_result) == 0:
            raise Exception(f"Unexpected Error in _is_evaluated function.")

        return existing_evaluation_qrels_count == updated_query_result[0]['count']


if __name__ == "__main__":
    db = Database()
    evaluator = Evaluator(db)
    language = "pt"
    top_k = 10

    for token_type in ["sentence_with_keywords"]:  # TODO: Tokenizer.token_types():
        for model_name in config.MODELS:
            ds = DenseSearcher(
                client=qdrant_client.QdrantClient(url=config.QDRANT_HOST, port=config.QDRANT_GRPC_PORT),
                model_name=model_name,
                token_type=token_type,
                language=language,
            )
            evaluator.evaluate(ds, top_k)

    ss = SparseSearcher(client=pysolr.Solr(config.SOLR_URL), top_k=top_k)
    evaluator.evaluate(ss, top_k)

    rs = RepositorySearcher(top_k=top_k)
    evaluator.evaluate(rs, top_k)
    #
    # evaluator.create_reports()
