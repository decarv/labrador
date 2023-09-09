"""
encoder

Copyright (c) 2023 Henrique Araújo de Carvalho

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

import sys
import os
import logging
import numpy as np
import pandas as pd

import config
from utils import log, save_embeddings, save_indices, embeddings_path
from sentence_transformers import SentenceTransformer

from utils import batch_generator


class Encoder:

    SEMANTIC_UNITS_TYPES = ["paragraphs", "sentences"]

    """TODO: Documentation"""
    def __init__(self, data, model_name: str,
                 embeddings_dir=None, indices_dir=None,
                 batch_size=64, language: str = "pt",
                 models_dir=None):
        self.data = data
        self.language = language
        self.model_name = model_name.split("/")[-1]
        self.model = SentenceTransformer(model_name, device='cuda', cache_folder=models_dir)
        self.indices_dir = indices_dir
        self.embeddings_dir = embeddings_dir
        if not os.path.exists(self.embeddings_dir) or not os.path.exists(self.indices_dir):
            logging.error("Either embeddings or indices save path does not exist.")
            sys.exit(1)
        self.batch_size = batch_size
        self.encoded_data = None

    @staticmethod
    def _clean_embedding_tokens(embedding_units: list[str]) -> list[str]:
        """
        Cleans embedding units.
        Removes leading and trailing whitespace from each string in the given list of strings and removes empty strings
        from the given list of strings.
        """
        embedding_units = [s.strip() for s in embedding_units]
        embedding_units = [s for s in embedding_units if s != '']
        return embedding_units

    @log
    def _generate_tokens(self, title: str, abstract: str, keywords: str) -> dict[str, list[str]]:
        """
        Generates textual groupings to serve as inputs for embeddings.

        Parameters:
            title (str): Document title.
            abstract (str): Document abstract.
            keywords (str): Document keywords.

        Returns:
            - dict: Dictionary containing:
                "sentences" -> List of sentences as units to be vectorized.
                "text" -> List containing the whole concatenated text.
                "sentences_and_text" -> Combined list of sentences and the entire text.
        """

        paragraphs_tokens: list[str] = [(title + ". " + abstract + " " + keywords).strip()]

        title_units: list[str] = title.split(".")
        abstract_units: list[str] = abstract.split(".")

        # Split keywords by uppercase letters
        keywords_units: list[str] = []
        keyword = ""
        for word in keywords.split():
            if word[0].isupper():
                if keyword != "":
                    keywords_units.append(keyword.strip())
                keyword = word
            else:
                keyword += " " + word
        keywords_units.append(keyword)

        sentences_tokens = self._clean_embedding_tokens(
            title_units + abstract_units + keywords_units
        )

        tokens_dict = {
            "paragraphs": paragraphs_tokens,
            "sentences": sentences_tokens,
        }
        assert list(tokens_dict.keys()) == self.SEMANTIC_UNITS_TYPES
        return tokens_dict

    @log
    def _structure_data_for_embedding(self, save=True):
        """
        TODO: Documentation
        - dict: Dictionary structured as:
            {
                "paragraphs/sentences": {
                    "indices": List mapping single encode input to the index in data.
                    "embedding_units": List of text units for encoding.
                }
            }
        """
        data_to_encode = {k: [] for k in self.SEMANTIC_UNITS_TYPES}
        indices = {k: [] for k in self.SEMANTIC_UNITS_TYPES}
        for index, row in self.data.iterrows():
            embedding_units = self._generate_tokens(
                row[f"title_{self.language}"],
                row[f"abstract_{self.language}"],
                row[f"keywords_{self.language}"]
            )
            for units_type in embedding_units.keys():
                data_to_encode[units_type] += embedding_units[units_type]
                indices[units_type] += [index for _ in range(len(embedding_units[units_type]))]

        if save and self.indices_dir:
            for units_type, ids in indices.items():
                save_indices(ids, self.indices_dir, units_type, self.language)

        return data_to_encode, indices

    @log
    def _encode_tokens(self, embedding_units) -> np.ndarray:
        """TODO: Documentation"""
        embeddings: list[np.ndarray] = []
        for batch in batch_generator(embedding_units):
            embedding = self.model.encode(batch, batch_size=len(batch), show_progress_bar=False)
            embeddings.append(embedding)
        return np.concatenate(embeddings)

    @log
    def encode(self, save=True):
        """TODO: Documentation"""
        data_for_embedding, indices = self._structure_data_for_embedding()
        self.encoded_data = {}
        for tokens_type, tokens in data_for_embedding.items():
            # Check if embeddings are cached
            if os.path.exists(embeddings_path(self.embeddings_dir, self.model_name, tokens_type, self.language)):
                continue
            embeddings = self._encode_tokens(tokens)
            self.encoded_data[tokens_type] = embeddings
            if save and self.embeddings_dir:
                save_embeddings(embeddings, self.embeddings_dir, self.model_name, tokens_type,
                                self.language)
        return self.encoded_data, indices


if __name__ == "__main__":
    metadata = pd.read_csv(os.path.join(config.DATA_DIR, "metadata.csv"), keep_default_na=False)
    for model in config.MODELS:
        encoder = Encoder(
            metadata,
            model_name=model,
            embeddings_dir=config.EMBEDDINGS_DIR,
            indices_dir=config.INDICES_DIR,
            models_dir=config.MODEL_CACHE_DIR
        )
        encoder.encode()
