"""
tokenizer

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
import gc
import os
from typing import Optional, Generator, Callable
from psycopg2.extras import RealDictRow

import utils

logger = utils.configure_logger(__name__)


class Tokenizer:
    def __init__(self, token_type: str, language: str = "pt"):
        _token_generator_name: str = f"_generate_{token_type}_tokens"
        if not hasattr(self, _token_generator_name):
            raise ValueError(f"Invalid token type: {token_type}. Valid token types are: {Tokenizer.token_types()}")

        self._tokenizer: Callable = getattr(self, _token_generator_name)
        self.token_type = token_type
        self.language = language
        self.path = utils.tokens_path(self.token_type, self.language)
        self.tokens: list[list[str]] = []

    def tokenize(self, data: Generator) -> list[list[str]]:
        if len(self.tokens) > 0:
            return self.tokens

        if os.path.exists(self.path):
            logger.info(f"Loaded tokens from {self.path}.")
            return utils.tokens_load(self.token_type, self.language)

        if isinstance(data, Generator):
            for value in data:
                if isinstance(value, list):
                    for instance in value:
                        self.tokens.append(self._tokenizer(instance))
                else:
                    self._tokenizer(value)
            self._save()
            return self.tokens

        raise NotImplementedError("Data must be a generator.")

    @classmethod
    def token_types(cls) -> list[str]:
        return [attr[10:-7] for attr in dir(cls) if attr.startswith("_generate_") and attr.endswith("_tokens")]

    def _save(self):
        utils.tokens_save(self.tokens, self.token_type, self.language)
        logger.info(f"Tokens generated and saved to {self.path}.")

    def clear_cache(self):
        del self.tokens
        gc.collect()
        self.tokens = []

    def _get_instance_tokens(self, instance: RealDictRow) -> tuple[list[str], list[str], list[str]]:
        tt: list[str] = instance[f"title_tokens_{self.language}"]
        at: list[str] = instance[f"abstract_tokens_{self.language}"]
        kt: list[str] = instance[f"keywords_tokens_{self.language}"]

        # TODO: Remove this when the data is fixed.
        if tt is None:
            tt = [""]
        if at is None:
            at = [""]
        if kt is None:
            kt = [""]

        return tt, at, kt

    def _generate_paragraph_tokens(self, instance: RealDictRow) -> list[str]:
        tt, at, kt = self._get_instance_tokens(instance)
        paragraph_tokens: list[str] = [". ".join(tt + at + kt)]
        return paragraph_tokens

    def _generate_sentence_tokens(self, instance: RealDictRow) -> list[str]:
        tu, au, ku = self._get_instance_tokens(instance)
        sentence_tokens: list[str] = tu + au + ku
        return sentence_tokens

    def _generate_sentence_with_keywords_tokens(self, instance: RealDictRow) -> list[str]:
        tt, at, kt = self._get_instance_tokens(instance)
        sentence_tokens: list[str] = tt + at
        kw_str: str = ". ".join(kt)
        for i in range(len(sentence_tokens)):
            sentence_tokens[i] += f". {kw_str}"
        return sentence_tokens

    def _generate_8gram_tokens(self, instance: RealDictRow) -> list[str]:
        n = 8
        tt, at, kt = self._get_instance_tokens(instance)
        ngrams_tt: list[str] = []
        ngrams_at: list[str] = []
        for i in range(len(tt)):
            ngrams_tt += self.__generate_ngram_tokens(tt[i].split(), n)
        for i in range(len(at)):
            ngrams_at += self.__generate_ngram_tokens(at[i].split(), n)
        return ngrams_tt + ngrams_at

    def _generate_8gram_with_keywords_tokens(self, instance: RealDictRow) -> list[str]:
        n = 8
        tt, at, kt = self._get_instance_tokens(instance)
        kw_str = ". ".join(kt)
        ngrams_tt: list[str] = []
        ngrams_at: list[str] = []
        for i in range(len(tt)):
            ngrams_tt += self.__generate_ngram_tokens(tt[i].split(), n, kw_str)
        for i in range(len(at)):
            ngrams_at += self.__generate_ngram_tokens(at[i].split(), n, kw_str)
        return ngrams_tt + ngrams_at

    @staticmethod
    def __generate_ngram_tokens(
            words: list[str], n: int, kw_tokens: Optional[str] = None
    ) -> list[str]:
        """
        Generates n-gram tokens from a list of words.
        """
        if len(words) < n:
            return words

        ngram_tokens: list[str] = []
        for i in range(len(words) - n + 1):
            ngram: str = " ".join(words[i:i + n])
            if kw_tokens:
                ngram += f". {kw_tokens}"
            ngram_tokens.append(ngram)
        return ngram_tokens
