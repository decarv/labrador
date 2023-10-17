"""
crawler

Copyright 2023 Henrique de Carvalho

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
from queue import Queue
from typing import Iterator

import chardet
import requests
from bs4 import BeautifulSoup
from psycopg2.extras import RealDictRow

import config
from util import log, database
from ingest.models import Metadata, Webpage

logger = log.configure_logger(__file__)
log = log.log(logger)


class Crawler:
    """
    TODO:
        - Dynamic Sitemap Parsing
        - Reconnection Logic: _session_setup() establishes new session, but robust network handling is required
            maybe add _session_cleanup() as well
        - Works not found:
            - https://www.teses.usp.br/teses/disponiveis/39/39132/tde-08072021-104642/pt-br.php
    """
    def __init__(self, **kwargs):
        self.session: requests.Session

    def run(self) -> None:
        self._session_setup()

        queue: Queue[Webpage] = Queue()  # Queue to crawl
        queued: set[Webpage] = set()
        self._preprocess(queue, queued)

        while not queue.empty():
            webpage: Webpage = queue.get()

            try:
                logger.info(f"Processing Webpage object: {webpage}.")
                webpage.process(self.session)
            except Exception as e:
                logger.error(f"Error while processing webpage: {webpage.url}")
                database.errors_insert(
                    f"Ex: {e} | Component: Crawler | Url: {webpage.url}"
                )
                continue

            if webpage.is_crawlable:
                if webpage.is_metadata:
                    metadata = Metadata(webpage)
                    try:
                        database.documents_insert(metadata)
                    except Exception as e:
                        logger.error(f"Unable to store metadata for {metadata.url} in database. Error: {e}")
                        database.errors_insert(
                            f"Ex: {e} | Component: Crawler | Url: {metadata.url}"
                        )
                        continue

                for hyperlink in webpage.children_hyperlinks:
                    child_webpage: Webpage = Webpage(hyperlink)
                    if child_webpage.is_crawlable and child_webpage not in queued:
                        try:
                            logger.info(f"Inserting in database: {webpage.url}")
                            database.webpages_insert(webpage)
                            queue.put(child_webpage)
                            queued.add(child_webpage)
                        except Exception as e:
                            logger.error(f"Could not insert in database: {child_webpage.url}")
                            database.errors_insert(
                                f"Ex: {e} | Component: Crawler | Url: {child_webpage.url}"
                            )

            database.webpages_crawled_update(webpage)

    def crawl_sitemap(self):
        logger.info("Started Sitemap Crawler")
        sitemap_urls = [
            'https://www.teses.usp.br/sitemap/sitemap01.xml',
            'https://www.teses.usp.br/sitemap/sitemap02.xml',
            'https://www.teses.usp.br/sitemap/sitemap03.xml',
            'https://www.teses.usp.br/sitemap/201905.xml',
            'https://www.teses.usp.br/sitemap/201906.xml',
            'https://www.teses.usp.br/sitemap/201907.xml',
            'https://www.teses.usp.br/sitemap/201908.xml',
            'https://www.teses.usp.br/sitemap/201909.xml',
            'https://www.teses.usp.br/sitemap/201910.xml',
            'https://www.teses.usp.br/sitemap/201911.xml',
            'https://www.teses.usp.br/sitemap/201912.xml',
            'https://www.teses.usp.br/sitemap/202001.xml',
            'https://www.teses.usp.br/sitemap/202002.xml',
            'https://www.teses.usp.br/sitemap/202003.xml',
            'https://www.teses.usp.br/sitemap/202004.xml',
            'https://www.teses.usp.br/sitemap/202005.xml',
            'https://www.teses.usp.br/sitemap/202006.xml',
            'https://www.teses.usp.br/sitemap/202007.xml',
            'https://www.teses.usp.br/sitemap/202008.xml',
            'https://www.teses.usp.br/sitemap/202009.xml',
            'https://www.teses.usp.br/sitemap/202010.xml',
            'https://www.teses.usp.br/sitemap/202011.xml',
            'https://www.teses.usp.br/sitemap/202012.xml',
            'https://www.teses.usp.br/sitemap/202101.xml',
            'https://www.teses.usp.br/sitemap/202102.xml',
            'https://www.teses.usp.br/sitemap/202103.xml',
            'https://www.teses.usp.br/sitemap/202104.xml',
            'https://www.teses.usp.br/sitemap/202105.xml',
            'https://www.teses.usp.br/sitemap/202106.xml',
            'https://www.teses.usp.br/sitemap/202107.xml',
            'https://www.teses.usp.br/sitemap/202108.xml',
            'https://www.teses.usp.br/sitemap/202109.xml',
            'https://www.teses.usp.br/sitemap/202110.xml',
            'https://www.teses.usp.br/sitemap/202111.xml',
            'https://www.teses.usp.br/sitemap/202112.xml',
            'https://www.teses.usp.br/sitemap/202201.xml',
            'https://www.teses.usp.br/sitemap/202202.xml',
            'https://www.teses.usp.br/sitemap/202203.xml',
            'https://www.teses.usp.br/sitemap/202204.xml',
            'https://www.teses.usp.br/sitemap/202205.xml',
            'https://www.teses.usp.br/sitemap/202206.xml',
            'https://www.teses.usp.br/sitemap/202207.xml',
            'https://www.teses.usp.br/sitemap/202208.xml',
            'https://www.teses.usp.br/sitemap/202209.xml',
            'https://www.teses.usp.br/sitemap/202210.xml',
            'https://www.teses.usp.br/sitemap/202211.xml',
            'https://www.teses.usp.br/sitemap/202212.xml',
            'https://www.teses.usp.br/sitemap/202301.xml',
            'https://www.teses.usp.br/sitemap/202302.xml',
            'https://www.teses.usp.br/sitemap/202303.xml',
            'https://www.teses.usp.br/sitemap/202304.xml',
            'https://www.teses.usp.br/sitemap/202305.xml',
            'https://www.teses.usp.br/sitemap/202306.xml'
        ]

        queue: Queue[Webpage] = Queue()  # Queue to crawl
        for url in sitemap_urls:
            try:
                response = self.session.get(url)
                if not response.ok:
                    raise Exception(f"Request Error @ url {url}. Status Code: {response.status_code}")
                encoding = chardet.detect(response.content)['encoding']
                response.encoding = encoding

                logger.info(f"Parsing url: {url}")
                xml_soup = BeautifulSoup(response.content, "xml")
            except Exception as e:
                logger.error(f"Error while requesting and parsing: {url}")
                database.errors_insert(
                    f"Ex: {e} | Component: Crawler | Url: {url}"
                )
                continue

            for child_url in (loc.text for loc in xml_soup.find_all("loc")):
                try:
                    webpage = Webpage(child_url)
                    if not database.webpages_instance_exists(webpage.url):
                        logger.info(f"Inserting in database: {webpage.url}")
                        database.webpages_insert(webpage)
                    else:
                        logger.info(f"Database already contains url: {child_url}. Continuing.")

                    if not database.metadata_instance_exists(webpage.url):
                        logger.info(f"Queueing: {child_url}")
                        queue.put(webpage)
                except Exception as e:
                    logger.error(f"Could not parse webpage: {child_url}")
                    database.errors_insert(
                        f"Ex: {e} | Component: Crawler | Url: {child_url}"
                    )

        while not queue.empty():
            webpage: Webpage = queue.get()
            try:
                logger.info(f"Processing Webpage object: {webpage}.")
                webpage.process(self.session)
            except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as e:
                logger.error(f"Error while requesting url: {webpage.url}. Error: {e}")
                database.errors_insert(
                    f"Ex: {e} | Component: Crawler | Url: {webpage.url}"
                )
                self._session_setup()
                webpage.delete_cache()
                queue.put(webpage)
                continue
            except Exception as e:
                logger.error(f"Error while processing webpage: {webpage.url}")
                database.errors_insert(
                    f"Ex: {e} | Component: Crawler | Url: {webpage.url}"
                )
                continue

            if webpage.is_metadata:
                try:
                    metadata: Metadata = Metadata(webpage)
                except Exception as e:
                    logger.error(f"Unable to parse metadata for {webpage.url}. Exception: {e}")
                    database.errors_insert(
                        f"Ex: {e} | Component: Crawler | Url: {webpage.url} | Consequence: Unable to parse."
                    )
                    continue

                try:
                    database.documents_insert(metadata)
                except Exception as e:
                    logger.error(f"Unable to store metadata for {metadata.url} in database. Error: {e}")
                    database.errors_insert(
                        f"Ex: {e} | Component: Crawler | Url: {webpage.url} | Consequence: Unable to parse."
                    )
                    continue

            database.webpages_crawled_update(webpage)

    def _session_setup(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept-Language": "pt-br,pt-BR"})

    def _preprocess(self, queue: Queue[Webpage], queued: set[Webpage]):
        if database.webpages_empty():
            logger.info("Database empty. Starting crawl from config.BASE_URL.")
            webpage = Webpage(config.BASE_URL)
            try:
                database.webpages_insert(webpage)
            except Exception as e:
                logger.error(f"Execute preprocess error: 'Could not insert to database': {e}")
                sys.exit(1)
            queue.put(webpage)
            queued.add(webpage)
        else:
            logger.info("Database not empty. Adding visited urls to visited set and to queue.")
            instances_batches: Iterator[list[RealDictRow]] = database.webpages_read()
            for batch in instances_batches:
                for instance in batch:
                    webpage = Webpage(instance['url'])
                    queued.add(webpage)
                    if not instance['is_crawled']:
                        queue.put(webpage)


if __name__ == "__main__":
    database.conn_pool_init(1, 10)
    crawler = Crawler()
    # crawler.run()
    # crawler.crawl_sitemap()
