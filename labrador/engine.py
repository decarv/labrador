"""
engine

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

from multiprocessing import Pool

from dense.tokenizer import Tokenizer
from dense.encoder import Encoder
from util import database
import config

language: str = "pt"
token_type: str = "sentence_with_keywords"
model_name: str = list(config.MODELS.keys())[1]

db: database.Database = database.Database()


class Engine:
    pass


def run_tokenizer():
    tokenizer: Tokenizer = Tokenizer(db)
    num_tokenizer_workers: int = 3
    min_id = 1
    max_id = db.select("""SELECT id FROM documents ORDER BY id DESC LIMIT 1;""")[0]['id']
    count = db.select("""SELECT COUNT(*) AS count FROM documents;""")[0]['count']
    limits = list(range(min_id, max_id, count // num_tokenizer_workers))
    id_ranges = [(a, b) for a, b in zip(limits[:-1], limits[1:])]

    with Pool(processes=num_tokenizer_workers) as pool:
        pool.map(tokenizer.loop, id_ranges)

def get_id_ranges(table, num_workers):
    ids = [row['id'] for row in db.select(
        f"""SELECT id FROM {table} WHERE token_type = %s ORDER BY id;""", (token_type,))]

    num_workers = min(len(ids), num_workers)

    id_ranges = []
    range_length = len(ids) // num_workers
    first: int = 0
    for i in range(num_workers):
        last = range_length * (i + 1)
        if last >= len(ids):
            last = -1
        id_ranges.append((ids[first], ids[last]))
        first = last + 1
    return id_ranges


def run_encoder():
    encoder: Encoder = Encoder(db, model_name)
    num_encoder_workers: int = 2

    id_ranges = get_id_ranges("tokens", num_encoder_workers)
    with Pool(processes=num_encoder_workers) as pool:
        pool.map(encoder.run, id_ranges)


if __name__ == "__main__":
    encoder: Encoder = Encoder(db, model_name)
    num_encoder_workers: int = 2

    id_ranges = get_id_ranges("tokens", num_encoder_workers)
    with Pool(processes=num_encoder_workers) as pool:
        pool.map(encoder.run, id_ranges)



