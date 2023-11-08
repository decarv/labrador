"""
crawler

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
import asyncio
import time
from queue import Queue
from typing import Optional

import chardet
import requests
from bs4 import BeautifulSoup

from util import log, database
from ingest.models import RawData, RawDataAsync

logger = log.configure_logger(__file__)
log = log.log(logger)


class Crawler:
    """
    TODO:
        - Dynamic sitemap.xml parsing: currently all the urls are parsed during one execution.
        - Caching sitemap urls
        - Consider using tenacity for retries
        - Consider using asyncio
    NOTES:
        - Works not found:
            - https://www.teses.usp.br/teses/disponiveis/39/39132/tde-08072021-104642/pt-br.php
    """
    def __init__(self):
        self.session: requests.Session
        self.sitemap_url = "https://teses.usp.br/sitemap/sitemap.xml"

    async def crawl_sitemap_async(self):
        self._setup()
        queue: Queue[str] = self._create_url_queue()
        while not queue.empty():
            raw_data_batch: list[RawData] = self._create_raw_data_batch(queue)
            try:
                await database.raw_data_batch_insert_async(raw_data_batch)
                logger.info(f"Successfully stored {len(raw_data_batch)} raw data objects in database.")
            except Exception as e:
                message = f"Uncaught Exception | {type(e)} | {e} | Component: Crawler"
                logger.error(message)
                database.errors_insert(message)
                for raw_data in raw_data_batch:
                    queue.put(raw_data.url)

    async def _crawl_sitemap_async(self):
        self._setup()
        queue: asyncio.Queue[str] = await self._create_url_queue_async()
        while not queue.empty():
            raw_data_batch: list[RawData] = self._create_raw_data_batch(queue)
            try:
                await database.raw_data_batch_insert_async(raw_data_batch)
                logger.info(f"Successfully stored {len(raw_data_batch)} raw data objects in database.")
            except Exception as e:
                message = f"Uncaught Exception | {type(e)} | {e} | Component: Crawler"
                logger.error(message)
                database.errors_insert(message)
                for raw_data in raw_data_batch:
                    await queue.put(raw_data.url)

    def _setup(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept-Language": "pt-br,pt-BR"})

    def _create_raw_data_batch(self, queue: Queue):
        insert_batch_size: int = 32
        raw_data_batch: list[RawData] = []
        while not queue.empty() and len(raw_data_batch) < insert_batch_size:
            url = queue.get()
            try:
                raw_data: RawData = self._create_raw_data(url)
                if raw_data is None:
                    queue.put(url)
                    continue
                raw_data_batch.append(raw_data)
            except RawData.RawDataError:
                logger.info(f"RawDataError: {url}")
                database.errors_insert(f"RawDataError: {url}")
        return raw_data_batch

    def _create_raw_data(self, url) -> Optional[RawData]:
        retries: int = 5
        while retries > 0:
            try:
                return RawData(url)
            except requests.HTTPError:
                logger.info(f"Failed to fetch {url} | Retries left: {retries}")
                self._setup()
                retries -= 1
            except requests.exceptions.ConnectionError:
                logger.info(f"Failed to fetch {url} | ConnectionError | Sleeping for 5 minutes")
                time.sleep(300)
                self._setup()
                retries -= 1
        logger.info(f"Failed to fetch {url} | Retries left: {retries}")
        return None

    async def _create_raw_data_batch_async(self, queue: Queue):
        insert_batch_size: int = 64
        raw_data_batch: list[RawDataAsync] = []
        while not queue.empty() and len(raw_data_batch) < insert_batch_size:
            url = queue.get()
            raw_data: RawDataAsync = await self._create_raw_data_async(url)
            if raw_data is None:
                queue.put(url)
                continue
            raw_data_batch.append(raw_data)
        return raw_data_batch

    async def _create_raw_data_async(self, url) -> Optional[RawDataAsync]:
        retries: int = 3
        while retries > 0:
            try:
                return RawDataAsync(url)
            except requests.HTTPError:
                logger.info(f"Failed to fetch {url} | Retries left: {retries}")
                self._setup()
                retries -= 1
        logger.info(f"Failed to fetch {url} | Retries left: {retries}")
        return None

    def _create_url_queue(self) -> Queue[str]:
        response = requests.get(self.sitemap_url)
        if response.status_code != 200:
            logger.info(f"Failed to fetch sitemap: {response.status_code}")
        encoding = chardet.detect(response.content)['encoding']
        response.encoding = encoding
        xml_soup = BeautifulSoup(response.content, "xml")
        inserted_urls = set(database.raw_data_get_urls())

        root_urls = [loc.text for loc in xml_soup.find_all("loc")][1:]

        queue: Queue[str] = Queue()
        for root_url in root_urls:
            response = requests.get(root_url)
            if response.status_code != 200:
                logger.info(f"Failed to fetch sitemap: {response.status_code}")
            encoding = chardet.detect(response.content)['encoding']
            response.encoding = encoding
            xml_soup = BeautifulSoup(response.content, "xml")
            urls = [loc.text for loc in xml_soup.find_all("loc")]
            for url in urls:
                if url in inserted_urls:
                    continue
                queue.put(url)

        return queue

    async def _create_url_queue_async(self) -> asyncio.Queue[str]:
        raise NotImplementedError


if __name__ == "__main__":
    crawler = Crawler()
    while True:
        asyncio.run(crawler.crawl_sitemap_async())
        time.sleep(60 * 60)
