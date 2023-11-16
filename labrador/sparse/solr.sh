
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field-type": {
    "name": "indexed_text",
    "class": "solr.TextField",
    "positionIncrementGap": "100",
    "indexAnalyzer": {
      "tokenizer": {
        "class": "solr.StandardTokenizerFactory"
      },
      "filters": [
        {"class": "solr.LowerCaseFilterFactory"},
        {"class": "solr.StopFilterFactory", "ignoreCase": true, "words": "stopwords.txt", "format": "wordset"},
        {"class": "solr.BrazilianStemFilterFactory"}
      ]
    },
    "queryAnalyzer": {
      "tokenizer": {
        "class": "solr.StandardTokenizerFactory"
      },
      "filters": [
        {"class": "solr.LowerCaseFilterFactory"},
        {"class": "solr.StopFilterFactory", "ignoreCase": true, "words": "stopwords.txt", "format": "wordset"},
        {"class": "solr.BrazilianStemFilterFactory"}
      ]
    }
  }
}' http://192.168.15.195:8983/solr/labrador/schema

# Add 'doc_id' field
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "doc_id",
    "type": "plong",
    "indexed": true,
    "stored": true
  }
}' http://192.168.15.195:8983/solr/labrador/schema

# Add 'author' field
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "author",
    "type": "text_general",
    "indexed": true,
    "stored": true
  }
}' http://192.168.15.195:8983/solr/labrador/schema

# Add 'title' field
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "title",
    "type": "indexed_text",
    "indexed": true,
    "stored": true
  }
}' http://192.168.15.195:8983/solr/labrador/schema

# Add 'abstract' field
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "abstract",
    "type": "indexed_text",
    "indexed": true,
    "stored": true
  }
}' http://192.168.15.195:8983/solr/labrador/schema

# Add 'keywords' field
curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field": {
    "name": "keywords",
    "type": "indexed_text",
    "indexed": true,
    "stored": true
  }
}' http://192.168.15.195:8983/solr/labrador/schema

curl "http://localhost:8983/solr/admin/cores?action=RELOAD&core=labrador"

