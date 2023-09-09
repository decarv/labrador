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


class Indexer:
    pass

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