from search.searcher import Searcher


class KeywordSearcher(Searcher):
    def __init__(self, *args, **kwargs):
        super(KeywordSearcher, self).__init__(*args, **kwargs)
        self.index = kwargs.get("index")
        if not self.index:
            raise ValueError("Missing required argument: index.")

        self.avgdl = sum(map(len, (doc['text'] for doc in self.index))) / len(self.index)
        self.N = len(self.index)
        self.k1 = 1.5
        self.b = 0.75

    def _retrieve(self, query: str, top_k: int) -> list[dict]:
        max_scores = []
        for document in self.index:
            max_scores = max(max_score, self.bm25(self.encode_query(query), document))
        return scores[:top_k]

    def bm25(self, encoded_query, document):
        score = 0.0
        doc_freqs = Counter(document['text'].split())
        for term in encoded_query:
            if term in doc_freqs:
                f = doc_freqs[term]
                n = sum(1 for doc in self.index if term in doc['text'])
                idf = math.log((self.N - n + 0.5) / (n + 0.5) + 1)
                score += (idf * f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * len(document['text'].split()) / self.avgdl))
        return score

    def encode_query(self, query):
        # Tokenization and case folding for the query
        return query.lower().split()