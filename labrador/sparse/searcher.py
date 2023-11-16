
import pysolr
from labrador.models import Searcher
from labrador.config import SOLR_URL


class SparseSearcher(Searcher):
    def __init__(self, *args, **kwargs):
        super(SparseSearcher, self).__init__(*args, **kwargs)
        self.client = kwargs.get('client')

    def _retrieve(self, query: str, top_k: int) -> list[dict]:
        results = self.client.search(
            q=f"abstract:{query}^1.5 OR title:{query}^2.0 OR keywords:{query}^1.5",
            rows=top_k,
        )
        hits = []
        for result in results:
            [result['doc_id']] = result['doc_id']  # TODO: fix this -> doc_id shouldn't be a list
            hits.append(result)
        return hits

    def _filter(self, hits, _filters=None):
        return hits

    def _rank(self, hits):
        return hits

    async def _retrieve_async(self, query: str, top_k: int) -> list[dict]:
        pass

    async def _filter_async(self, hits, _filters=None):
        pass

    async def _rank_async(self, hits):
        pass


if __name__ == '__main__':
    query = "ataque de negação de serviço"
    searcher = SparseSearcher(client=pysolr.Solr(SOLR_URL))
    hits = searcher.search(query, top_k=10)
    print(hits)
