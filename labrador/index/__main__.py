
import time
from multiprocessing import Pool

from tokenizer import Tokenizer
from encoder import Encoder
from util import database
import config

language: str = "pt"
token_type: str = "sentence_with_keywords"
model_name: str = list(config.MODELS.keys())[0]

db: database.Database = database.Database()


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
        pool.map(encoder.loop, id_ranges)


if __name__ == "__main__":
    # run_tokenizer()
    # encoder: Encoder = Encoder(db, model_name)
    # encoder.loop()
    # run_encoder()
