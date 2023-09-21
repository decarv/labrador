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
import gc
import json
import os
import time
from typing import Optional, Iterator

from psycopg2.extras import RealDictRow
import torch

import config

from encoder import Encoder

import utils
from tokenizer import Tokenizer
from searcher import LocalSearcher  # , DenseSearcher
import datetime

logger = utils.configure_logger(__name__)

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
        "busca semântica",
        "recuperação de informação",
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

def experiments_path(experiment_name: str, save_dir: str = config.RESULTS_DIR):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dt = datetime.datetime.now().strftime("%Y%m%d%H%M")
    filename = f"{dt}_{experiment_name}.json"
    path: str = os.path.join(save_dir, filename)
    return path


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

@utils.log(logger)
def local_searcher_pipeline(
        collection_name="abstracts", units_type="sentences", language="pt", device="cuda"
) -> tuple[list[float], list[list[dict]]]:
    # processor = Processor()

    local_searcher = LocalSearcher(
        model_name=config.MODELS[0],
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
    chunk_size: int = 1024 * 4
    language = "pt"
    token_types = ["sentence", "paragraph", "sentence_with_keywords"]# Tokenizer.token_types()
    for token_type in token_types:
        tokenizer = Tokenizer(token_type, language)
        for model in config.MODELS:
            token_batch_iterator: Iterator[list[str]] = tokenizer.tokens_generator()

            logger.info(f"CHECKPOINT: Starting encoding for {model}...")
            encoder: Encoder = Encoder(model_name=model, token_type=token_type)
            encoder.encode(token_batch_iterator)

