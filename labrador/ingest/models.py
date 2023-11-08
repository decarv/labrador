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
import bs4
import requests
from typing import Optional


class RawData:

    class RawDataError(Exception):
        pass

    def __init__(self, url: str):
        self.url: Optional[str] = None
        self.doi: Optional[str] = None
        self.type: Optional[str] = None
        self.author: Optional[str] = None
        self.institute: Optional[str] = None
        self.knowledge_area: Optional[str] = None
        self.committee: Optional[str] = None
        self.title_pt: Optional[str] = None
        self.title_en: Optional[str] = None
        self.keywords_pt: Optional[str] = None
        self.keywords_en: Optional[str] = None
        self.abstract_pt: Optional[str] = None
        self.abstract_en: Optional[str] = None
        self.publish_date: Optional[str] = None

        document_url_pattern: str = r"tde-\d+-\d+"
        is_metadata = re.search(document_url_pattern, url) is not None
        if not is_metadata:
            raise RawData.RawDataError("RawData is not a metadata page")
        self.parse_metadata(url)

    def __repr__(self):
        return str(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__.values())

    def parse_metadata(self, url: str):
        response = requests.get(url)
        if response.ok:
            html = response.content
        else:
            raise requests.HTTPError(f"Bad request: Status Code: {response.status_code}")

        soup = bs4.BeautifulSoup(html, "html.parser")
        self.url = url

        raw_metadata = soup.find_all(class_="DocumentoTexto")
        raw_metadata_keys = soup.find_all(class_="DocumentoTituloTexto")
        metadata = {
            k.text.strip().lower(): re.sub(r"\s+", " ", v.text.strip())
            for (k, v) in zip(raw_metadata_keys, raw_metadata)
        }
        self.doi = metadata.get("doi", None)
        self.type = metadata.get("documento", None)
        self.author = metadata.get("autor", None)
        self.institute = metadata.get("unidade da usp", None)
        self.knowledge_area = metadata.get("área do conhecimento", None)
        self.committee = metadata.get("banca examinadora", None)
        self.title_pt = metadata.get("título em português", None)
        self.keywords_pt = metadata.get("palavras-chave em português", None)
        self.title_en = metadata.get("título em inglês", None)
        self.keywords_en = metadata.get("palavras-chave em inglês", None)
        self.publish_date = metadata.get("data de publicação", None)

        raw_data = soup.find_all(class_="DocumentoTextoResumo")
        raw_data_keys = soup.find_all(class_="DocumentoTituloTexto2")
        data = {
            k.text.strip().lower(): re.sub(r"\s+", " ", v.text.strip())
            for (k, v) in zip(raw_data_keys, raw_data)
        }
        self.abstract_pt = data.get("resumo em português", None)
        self.abstract_en = data.get("resumo em inglês", None)

    def __hash__(self):
        return hash(self.doi)


class RawDataAsync:
    def __init__(self, url: str):
        raise NotImplementedError("RawDataAsync is not implemented yet")
