from labrador.models import Searcher

import lucene
import math

from org.apache.lucene.analysis.br import BrazilianAnalyzer
from org.apache.lucene.document import Document, Field, TextField
from org.apache.lucene.index import DirectoryReader, IndexWriter, IndexWriterConfig
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher

from org.apache.pylucene.store import PythonDirectory


# Create a directory to store the index
INDEX_DIR = os.path.join(DATA_DIR, "sparse.index")

# Initialize lucene and JVM
lucene.initVM()

# Create an in-memory index
PythonDirectory(os.path.join(os.getcwd(), "index"))

# Configure a Brazilian analyzer
analyzer = BrazilianAnalyzer()

# Set up an IndexWriter configuration
config = IndexWriterConfig(analyzer)
config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)

# Create the IndexWriter
writer = IndexWriter(config)

# List of documents in Portuguese to index
documents = [
    "A rápida raposa marrom pula sobre o cão preguiçoso",
    "Lucene é um motor de busca de texto completo em Java",
    "PyLucene é uma extensão Python para Lucene"
]

# Adding documents to the index
for content in documents:
    doc = Document()
    doc.add(TextField("content", content, Field.Store.YES))
    writer.addDocument(doc)

# Commit everything and close the writer
writer.commit()
writer.close()


# Function to search the index
def search(query_string, analyzer, index):
    # Parse the query
    query = QueryParser("content", analyzer).parse(query_string)

    # Create a searcher
    reader = DirectoryReader.open(index)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(lucene.search.similarities.BM25Similarity())

    # Perform the search
    hits = searcher.search(query, 10).scoreDocs

    # Iterate through the results
    for hit in hits:
        doc = searcher.doc(hit.doc)
        content = doc.get("content")
        print(f"Documento: {content}, Pontuação: {hit.score}")

    # Close the reader
    reader.close()


# Search for 'raposa' (fox in Portuguese)
search_query = "raposa"
search(search_query, analyzer, index)


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