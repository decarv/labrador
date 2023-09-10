"""
encoder

Copyright 2023 Henrique Araújo de Carvalho

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

import torch
from sentence_transformers import SentenceTransformer

import config
import utils
from utils import log

logger = utils.configure_logger(__name__)


class Encoder:

    def __init__(self, model_name: str, language: str = "pt"):
        self.language = language
        self.model_name = model_name.split("/")[-1]
        self.model = SentenceTransformer(
            model_name, device='cuda', cache_folder=config.MODEL_CACHE_DIR
        )

    @log(logger)
    def encode(self, tokens: list[list[str]]) -> tuple[torch.tensor, list[int]]:
        logger.info(f"Encoding {len(tokens)} tokens.")
        flattened_tokens: list[str] = []
        indices: list[int] = []
        for i in range(len(tokens)):
            flattened_tokens += tokens[i]
            indices += [i] * len(tokens[i])
        embeddings: torch.tensor = self._encode_tokens(flattened_tokens)
        return embeddings, indices

    @log(logger)
    def _encode_tokens(self, tokens: list[str]) -> torch.tensor:
        embeddings: list[torch.tensor] = []
        for batch in self.batch_generator(tokens):
            embedding: torch.tensor = self.model.encode(batch, batch_size=len(batch), convert_to_tensor=True)
            embeddings.append(embedding)
        return torch.cat(embeddings)
    #
    # def _tokenize_instance(self, instance: RealDictRow, token_type: str) -> list[str]:
    #     match token_type:
    #         case "paragraph":
    #             return self._get_paragraph_tokens(instance)
    #         case "sentence":
    #             return self._get_sentence_tokens(instance)
    #         case "sentence_with_keywords":
    #             return self._generate_sentence_with_keywords_tokens(instance)
    #         case "8gram":
    #             return self._generate_8gram_tokens(instance)
    #         case "8gram_with_keywords":
    #             return self._generate_8gram_tokens(instance)
    #         case _:
    #             raise ValueError(f"Invalid token type: {token_type}. Valid token types are: {self.TOKEN_TYPES}")
    #
    # def _get_instance_tokens(self, instance: RealDictRow) -> tuple[list[str], list[str], list[str]]:
    #     tt: list[str] = instance[f"title_tokens_{self.language}"]
    #     at: list[str] = instance[f"abstract_tokens_{self.language}"]
    #     kt: list[str] = instance[f"keywords_tokens_{self.language}"]
    #
    #     return tt, at, kt
    #
    # def _get_paragraph_tokens(self, instance: RealDictRow) -> list[str]:
    #     tt, at, kt = self._get_instance_tokens(instance)
    #     paragraph_tokens: list[str] = [" ".join(tt + at + kt)]
    #     return paragraph_tokens
    #
    # def _get_sentence_tokens(self, instance: RealDictRow) -> list[str]:
    #     tu, au, ku = self._get_instance_tokens(instance)
    #     sentence_tokens: list[str] = tu + au + ku
    #     return sentence_tokens
    #
    # def _generate_sentence_with_keywords_tokens(self, instance: RealDictRow) -> list[str]:
    #     tt, at, kt = self._get_instance_tokens(instance)
    #     sentence_tokens: list[str] = tt + at
    #     kt_str: str = " ".join(kt)
    #     for token in sentence_tokens:
    #         token += " " + " ".join(kt_str)
    #     return sentence_tokens
    #
    # def _generate_8gram_tokens(self, instance: RealDictRow) -> list[str]:
    #     tt, at, kt = self._get_instance_tokens(instance)
    #     return self._generate_ngram_tokens(tt, 8) + self._generate_ngram_tokens(tt, 8)
    #
    # def _generate_8gram_tokens_with_keywords(self, instance: RealDictRow) -> list[str]:
    #     tt, at, kt = self._get_instance_tokens(instance)
    #     return self._generate_ngram_tokens(tt, 8, kt) + self._generate_ngram_tokens(tt, 8, kt)
    #
    # @staticmethod
    # def _generate_ngram_tokens(
    #         words: list[str], n: int, kw_tokens: Optional[list[str]] = None
    # ) -> list[str]:
    #     """
    #     Generates n-gram tokens from a list of words.
    #     """
    #     if len(words) < n:
    #         return words
    #
    #     ngram_tokens: list[str] = []
    #     for i in range(len(words) - n + 1):
    #         ngram: str = " ".join(words[i:i + n])
    #         if kw_tokens:
    #             ngram += " " + " ".join(kw_tokens)
    #         ngram_tokens.append(ngram)
    #     return ngram_tokens

    @staticmethod
    def batch_generator(data, batch_size=64):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
