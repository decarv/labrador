import asyncio
import time

import bs4
import httpx
import requests

from labrador import config
from labrador.dense.processor import Processor
from labrador.models import Searcher
from labrador.util.log import logger


class RepositorySearcher(Searcher):
    def __init__(self, *args, **kwargs):
        super(RepositorySearcher, self).__init__()
        self.db = kwargs.get('database')
        if not self.db:
            raise ValueError("Missing required argument: database.")

    def _retrieve(self, query: str, top_k):
        base_url: str = config.THESES_QUERY_URL.format(query.replace(" ", "%20"))
        page = 1
        documents_urls: list[str] = []
        results: list[str] = []
        while len(results) < top_k:
            url: str = base_url + str(page)
            response: requests.Response = requests.get(url)
            if response is not None:
                soup = bs4.BeautifulSoup(response.content, 'html.parser')
                divs = soup.find_all(
                    "div",
                    class_=["dadosLinha dadosCor1", "dadosLinha dadosCor2"]
                )
                documents_urls += [div.a['href'] for div in divs]
            paths: list[str] = [Processor.extract_path_suffix(url) for url in documents_urls]
            results.extend(self.db.select(
                """SELECT d.id as doc_id, d.title_pt as title, d.abstract_pt as abstract, d.keywords_pt as keywords, 
                          d.author, d.url
                     FROM documents as d 
                    WHERE d.url_path_suffix = ANY(%s);""", var_args=(paths,)))
            page += 1
        return results


    async def _retrieve_async(self, query: str, top_k) -> list[dict]:
        base_url: str = config.THESES_QUERY_URL.format(query.replace(" ", "%20"))
        page = 1
        documents_urls: list[str] = []

        while len(documents_urls) < top_k:
            url: str = base_url + str(page)
            response: requests.Response = requests.get(url)  # TODO: aiohttp this later
            if response is not None:
                soup = bs4.BeautifulSoup(response.content, 'html.parser')
                divs = soup.find_all(
                    "div",
                    class_=["dadosLinha dadosCor1", "dadosLinha dadosCor2"]
                )
                documents_urls += [div.a['href'] for div in divs]
            page += 1

        paths: list[str] = [Processor.extract_path_suffix(url) for url in documents_urls]
        results: list[dict] = await self.db.select(
            """SELECT d.id as doc_id, d.title_pt as title, d.abstract_pt as abstract, d.keywords_pt as keywords, 
                      d.author, d.url
                 FROM documents as d 
                WHERE d.url_path_suffix = ANY(%s);""", var_args=(paths,))

        return results

    @staticmethod
    async def multi_fetch_async(urls: list[str]):
        async with httpx.AsyncClient(timeout=15.0) as client:
            if len(urls) == 1:
                return [await client.get(urls[0])]
            responses = await asyncio.gather(*[client.get(url) for url in urls])
            return responses

    def _rank(self, hits):
        return hits

    def _filter(self, hits, _filters=None):
        return hits

    async def _rank_async(self, hits):
        return hits

    async def _filter_async(self, hits, _filters=None):
        return hits


if __name__ == '__main__':
    from labrador.util.database import Database

    query = "aprendizagem de máquina com iot"
    start = time.time()
    searcher = RepositorySearcher(database=Database())
    results = searcher.search(query, top_k=10)
    print(len(results), results)

    print(f"Total time: {time.time() - start}")