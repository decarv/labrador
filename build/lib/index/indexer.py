"""
indexer

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
from psycopg2.extras import RealDictCursor

import util.database
from util import utils
from index import tokenizer
from util import log

logger = log.configure_logger(__file__)
log = log.log(logger)


class Indexer:
    # TODO: I can compress these indices maps with simple math
    def __init__(self, token_type: str, language: str):
        self.indices_map = None
        self.token_indices_map = None
        self.tokens = None
        self.token_type = token_type
        self.language = language

    @log
    def generate_index(self, keep_in_memory: bool = False) -> None:
        query = "SELECT data_id, tokens FROM tokens WHERE token_type = %s ORDER BY data_id;"
        indices_map = []
        token_indices_map = []
        with util.database.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (self.token_type,))
                instances = cursor.fetchall()
                for instance in instances:
                    tokens = instance['tokens']
                    indices_map.extend([instance['clean_metadata_id']] * len(tokens))
                    token_indices_map.extend(list(range(len(tokens))))

        if keep_in_memory:
            self.tokens = tokens
            self.token_indices_map = token_indices_map
            self.indices_map = indices_map
        utils.indices_save((indices_map, token_indices_map), self.token_type, self.language)


if __name__ == '__main__':
    for token_type in tokenizer.Tokenizer.token_types():
        for language in ['pt', 'en']:
            indexer = Indexer(token_type, language)
            indexer.generate_index(keep_in_memory=True)



# from annoy import AnnoyIndex
# search_index = AnnoyIndex(embeds.shape[1], 'angular')
# # Add all the vectors to the search index
# for i in range(len(embeds)):
#     search_index.add_item(i, embeds[i])
#
# search_index.build(10) # 10 trees
# search_index.save('test.ann')
# pd.set_option('display.max_colwidth', None)
#
#
# def search(query):
#     # Get the query's embedding
#     query_embed = co.embed(texts=[query]).embeddings
#
#     # Retrieve the nearest neighbors
#     similar_item_ids = search_index.get_nns_by_vector(query_embed[0],
#                                                       3,
#                                                       include_distances=True)
#     # Format the results
#     results = pd.DataFrame(data={'texts': texts[similar_item_ids[0]],
#                                  'distance': similar_item_ids[1]})
#
#     print(texts[similar_item_ids[0]])
#
#     return results
