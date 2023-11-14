
import psycopg
from psycopg.rows import dict_row
from util import database as db
from config import POSTGRESQL_URL


def verify_tokens_integrity():
    conn = psycopg.connect(POSTGRESQL_URL, row_factory=dict_row)
    cursor = conn.cursor()
    cursor.execute("SELECT unique_hash FROM tokens")
    tokens = cursor.fetchall()

    new_token_hashes = set()
    for token in tokens:
        new_token_hashes.add(token['unique_hash'])

    old_dbname = POSTGRESQL_URL.split()
    old_dbname[0] = "dbname=mobus"
    old_dbname = " ".join(old_dbname)

    conn2 = psycopg.connect(old_dbname, row_factory=dict_row)
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT id, unique_hash, token FROM tokens WHERE token_type = 'sentence_with_keywords'")
    old_tokens = cursor2.fetchall()
    inconsistencies = []

    for old_token in old_tokens:
        if old_token['unique_hash'] not in new_token_hashes:
            inconsistencies.append(old_token)
            db.errors_insert(f"Token {old_token['id']} not found in new database")

    for inc in inconsistencies:
        print(f"-> {inc}")
        print()

    print(f"Total inconsistencies: {len(inconsistencies)}")



if __name__ == '__main__':
    verify_tokens_integrity()