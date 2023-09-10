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

from typing import Optional, List, Tuple
from psycopg2.extras import RealDictRow


class Tokenizer:
    """
    Tokenization component for text.
    Forms of tokenization:
     a) paragraph tokenization;
     b) sentence tokenization;
     c) n-gram tokenization.

    These can be with or without keywords.
    """

    TOKEN_TYPES = ["paragraph", "sentence", "sentence_with_keywords", "8gram", "8gram_with_keywords"]

    def __init__(self, data: list[RealDictRow], language: str = "pt"):
        self.data = data
        self.language = language
        self.tokens: dict[str, list[list[str]]] = {k: [] for k in self.TOKEN_TYPES}

    def tokenize(self, token_type: str) -> list[list[str]]:
        if len(self.tokens[token_type]) > 0:
            return self.tokens[token_type]

        for instance in self.data:
            self.tokens[token_type].append(self._tokenize_instance(instance, token_type))
        return self.tokens[token_type]

    def _tokenize_instance(self, instance: RealDictRow, token_type: str) -> list[str]:
        match token_type:
            case "paragraph":
                return self._get_paragraph_tokens(instance)
            case "sentence":
                return self._get_sentence_tokens(instance)
            case "sentence_with_keywords":
                return self._generate_sentence_with_keywords_tokens(instance)
            case "8gram":
                return self._generate_8gram_tokens(instance)
            case "8gram_with_keywords":
                return self._generate_8gram_tokens(instance)
            case _:
                raise ValueError(f"Invalid token type: {token_type}. Valid token types are: {self.TOKEN_TYPES}")

    def _get_instance_tokens(self, instance: RealDictRow) -> tuple[list[str], list[str], list[str]]:
        tt: list[str] = instance[f"title_tokens_{self.language}"]
        at: list[str] = instance[f"abstract_tokens_{self.language}"]
        kt: list[str] = instance[f"keywords_tokens_{self.language}"]

        return tt, at, kt

    def _get_paragraph_tokens(self, instance: RealDictRow) -> list[str]:
        tt, at, kt = self._get_instance_tokens(instance)
        paragraph_tokens: list[str] = [" ".join(tt + at + kt)]
        return paragraph_tokens

    def _get_sentence_tokens(self, instance: RealDictRow) -> list[str]:
        tu, au, ku = self._get_instance_tokens(instance)
        sentence_tokens: list[str] = tu + au + ku
        return sentence_tokens

    def _generate_sentence_with_keywords_tokens(self, instance: RealDictRow) -> list[str]:
        tt, at, kt = self._get_instance_tokens(instance)
        sentence_tokens: list[str] = tt + at
        kt_str: str = " ".join(kt)
        for token in sentence_tokens:
            token += " " + " ".join(kt_str)
        return sentence_tokens

    def _generate_8gram_tokens(self, instance: RealDictRow) -> list[str]:
        tt, at, kt = self._get_instance_tokens(instance)
        return self._generate_ngram_tokens(tt, 8) + self._generate_ngram_tokens(tt, 8)

    def _generate_8gram_tokens_with_keywords(self, instance: RealDictRow) -> list[str]:
        tt, at, kt = self._get_instance_tokens(instance)
        return self._generate_ngram_tokens(tt, 8, kt) + self._generate_ngram_tokens(tt, 8, kt)

    @staticmethod
    def _generate_ngram_tokens(
            words: list[str], n: int, kw_tokens: Optional[list[str]] = None
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
                ngram += " " + " ".join(kw_tokens)
            ngram_tokens.append(ngram)
        return ngram_tokens
