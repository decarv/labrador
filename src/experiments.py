"""
experiments

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

import json
import numpy as np
import pandas as pd
import os
import sys
import time
import config
from typing import Union
import sqlite3
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import logging
from processor import Processor
from typing import Optional

import utils
from utils import log
import functools
from searcher import KeywordSearcher, LocalSearcher, DenseSearcher

QUERIES = [
    "ddos attack",
    "distributed denial of service attack",
    "ataque de negação de serviço distribuído",
    "ataque de negação de serviço distribuído com machine learning",
    "ataque de negação de serviço distribuído com aprendizagem de máquina",
]

queries = {
    "pt": [
        "ataque ddos",
        "ataque ddos com machine learning",
        "entendimento popular sobre dengue",
        "computação quântica",
        "políticas públicas sobre inteligência artificial",
        "sindicatos e atribuições sociais",
        "busca semântica"
    ],
    "en": [
        "ddos attack",
        "ddos attack with machine learning",
        "popular understanding of dengue",
        "quantum computing",
        "public policies about artificial inteligence",
        "unions and their social attributions",
        "semantic search"
    ]
}

ANSWERS_IDS = [
    ""
]


# Experiment functions

def run_experiment(name, description, pipeline, args, filename):
    times, outputs = pipeline(*args)
    with open(filename, 'w') as file:
        json.dump({
            'name': name,
            'description': description,
            'times': times,
            'outputs': outputs,
        }, file)


def read_experiment(filename):
    with open(filename, 'r') as file:
        return json.load(file)


# Evaluation functions

def evaluate_relevance(results, answers):
    """TODO: calculate DCG"""
    pass


def evaluate_speed(times):
    pass


# Pipelines
#
# def kw_searcher_pipeline() -> tuple[list[float], dict[str, list[dict]]]:
#     # processor = Processor()
#     kw_searcher = KeywordSearcher(
#         language="pt",
#         data=utils.load_metadata_from_csv()
#     )
#     results: dict[str, list[dict]] = {}
#     times: list[float] = []
#     for query in QUERIES:
#         # query = processor.process_text(query)
#         start = time.time()
#         results[query] = kw_searcher.search(query)
#         end = time.time()
#         times.append(end - start)
#     return times, results
#

def local_searcher_pipeline(
        collection_name="abstracts", units_type="sentences", language="pt", device="cuda"
) -> tuple[list[float], list[list[dict]]]:
    # processor = Processor()

    model = config.MODELS[0]
    local_searcher = LocalSearcher(
        collection_name=collection_name,
        device=device,
        language=language,
        units_type=units_type,
    )
    results: list[list[dict]] = []
    times: list[float] = []
    for query in QUERIES:
        # query = processor.process_query(query)
        start = time.time()
        results.append(local_searcher.search(query))
        end = time.time()
        times.append(end - start)
    return times, results


if __name__ == '__main__':
    run_experiment(
        name="Local Searcher",
        description="This experiment evaluates the performance of the local searcher.",
        pipeline=local_searcher_pipeline,
        args=(),
        filename=utils.experiment_results_path("local_searcher")
    )

