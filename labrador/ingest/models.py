"""
models

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

import re
import urllib
from typing import Optional

import requests
from bs4 import BeautifulSoup

import config


class Webpage:
    def __init__(self, link: str):

        self.url: str = link
        self.is_crawlable: bool = False
        self.html: Optional[bytes] = None
        self.soup: Optional[BeautifulSoup] = None
        self.is_document: bool = False
        self.is_metadata: bool = False
        self.children_hyperlinks: list = []

        self.is_processed: bool = False

        self._preprocess(link)

    def _preprocess(self, link):
        if link[0:2] == "//":
            link = link[2:]

        # TODO: self.parsed_url does not need to be an object variable
        self.parsed_url: urllib.parse.ParseResult = urllib.parse.urlparse(link)

        # Ignore external webpages
        if self.parsed_url.netloc not in config.BASE_URL or "".join(self.parsed_url.netloc.split("www.")) not in config.BASE_URL:
            self.is_crawlable = False
        # Ignore webpages in other languages
        elif "lang" in self.parsed_url.query and "lang=pt" not in self.parsed_url.query:
            self.is_crawlable = False
        # Ignore hrefs that are not urls
        elif self.parsed_url.scheme and self.parsed_url.scheme != "http" and self.parsed_url.scheme != "https":
            self.is_crawlable = False
        # All the good urls fall here
        else:
            self.url = urllib.parse.urljoin(config.BASE_URL, link)

            # Clean some bad urls
            if self.url.endswith("pt-br.php"):
                self.url = self.url.removesuffix("pt-br.php")

            self.parsed_url = urllib.parse.urlparse(self.url)

            webfiles = [".html", ".php"]
            document_pattern: str = r"\.[A-z|0-9]+$"
            document_pattern_search = re.search(document_pattern, self.url)
            self.is_document = document_pattern_search is not None and document_pattern_search.group() not in webfiles

            metadata_pattern: str = r"tde-\d+-\d+"
            self.is_metadata = re.search(metadata_pattern, self.url) is not None and not self.is_document

            self.is_crawlable = not self.is_document

    def process(self, session: Optional[requests.Session] = None) -> None:
        self._get_html()
        if self.html:
            self._get_soup()
            self._get_children_hyperlinks()
        self.is_processed = True

    def _get_children_hyperlinks(self):
        a_tags = self.soup.find_all("a")
        for a_tag in a_tags:
            hyperlink: str = a_tag.attrs.get("href")
            if hyperlink is not None:
                self.children_hyperlinks.append(hyperlink)

    def _get_html(self, session: Optional[requests.Session] = None) -> None:
        try:
            if session is not None:
                response = session.get(self.url)
            else:
                response = requests.get(self.url)
            if response.ok:
                self.html = response.content
            else:
                raise Exception(f"Bad request: Status Code: {response.status_code}")
        except Exception as e:
            raise e

    def _get_soup(self):
        try:
            self.soup = BeautifulSoup(self.html, "html.parser")
        except Exception as e:
            raise e

    @staticmethod
    def remove_lang_from_url(url):
        return re.sub(r'/\?&lang.*$', '', url)

    @staticmethod
    def extract_path_suffix(url):
        pattern = r"disponiveis\/(.*\/td.*\d)"
        match = re.search(pattern, url)
        path_suffix = match.group()
        return path_suffix

    def delete_cache(self):
        self.html = None
        self.soup = None

    def __hash__(self):
        if self.html:
            return hash(self.html)
        return hash(self.url)

    def __eq__(self, other):
        if not isinstance(other, Webpage):
            raise TypeError(f"Comparing Webpage object with {type(other)}")
        return self.url == other.url

    def __repr__(self):
        return f"{self.url}"


class Metadata:
    class MetadataError(Exception):
        pass

    def __init__(self, webpage: Webpage = None):
        self.url: str
        self.doi: str
        self.type: str
        self.author: str
        self.institute: str
        self.knowledge_area: str
        self.committee: str
        self.title_pt: str
        self.title_en: str
        self.keywords_pt: str
        self.keywords_en: str
        self.abstract_pt: str
        self.abstract_en: str
        self.publish_date: str

        if webpage is not None:
            self.parse_metadata(webpage)

    def __repr__(self):
        return str(self.__dict__)

    def parse_metadata(self, webpage):
        """
        TODO: documentation
            - add data cleaning
        :param webpage:
        :return:
        """
        if not webpage.is_processed:
            raise Metadata.MetadataError("Webpage must be processed before parsing metadata")

        try:
            self.url = webpage.url
            raw_metadata = webpage.soup.find_all(class_="DocumentoTexto")
            raw_metadata_keys = webpage.soup.find_all(class_="DocumentoTituloTexto")
            metadata = {
                k.text.strip().lower(): re.sub(r"\s+", " ", v.text.strip())
                for (k, v) in zip(raw_metadata_keys, raw_metadata)
            }
            self.doi = metadata.get("doi", None)
            self.type = metadata.get("documento", None)
            self.author = metadata.get("autor", None) # TODO: add data cleaning "Catalogo USP"



            self.institute = metadata.get("unidade da usp", None)
            self.knowledge_area = metadata.get("área do conhecimento", None)
            self.committee = metadata.get("banca examinadora", None)
            self.title_pt = metadata.get("título em português", None)
            self.keywords_pt = metadata.get("palavras-chave em português", None)
            self.title_en = metadata.get("título em inglês", None)
            self.keywords_en = metadata.get("palavras-chave em inglês", None)
            self.publish_date = metadata.get("data de publicação", None)

            raw_data = webpage.soup.find_all(class_="DocumentoTextoResumo")
            raw_data_keys = webpage.soup.find_all(class_="DocumentoTituloTexto2")
            data = {
                k.text.strip().lower(): re.sub(r"\s+", " ", v.text.strip())
                for (k, v) in zip(raw_data_keys, raw_data)
            }
            self.abstract_pt = data.get("resumo em português", None)
            self.abstract_en = data.get("resumo em inglês", None)
            return self
        except Exception as e:
            raise e

    def parse_db_instance(self, instance):
        """TODO: documentation"""
        attributes = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for inst, attr in zip(instance, attributes):
            setattr(self, attr, inst)
        return self

    def to_dict(self):
        """TODO: documentation
        attributes = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
            return {attr: getattr(self, attr) for attr in attributes}
        """
        return {
            "title_pt": self.title_pt,
            "title_en": self.title_en,
            "author": self.author,
            "abstract_pt": self.abstract_pt,
            "abstract_en": self.abstract_en,
            "url": self.url
        }

    def __hash__(self):
        return hash(self.doi)

