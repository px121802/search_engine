import os
import pickle
import math
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import time

TERM_INDEX_DIR = "term_index"
TOTAL_DOCS = 56000  # Replace with your final indexed doc count
stemmer = PorterStemmer()

def tokenize_query(query):
    return [stemmer.stem(token.lower()) for token in word_tokenize(query) if token.isalnum()]

def get_index_file(term):
    first = term[0].lower()
    if first in string.ascii_lowercase:
        return os.path.join(TERM_INDEX_DIR, f"{first}.pkl")
    return os.path.join(TERM_INDEX_DIR, "other.pkl")

def load_postings(term):
    index_file = get_index_file(term)
    if not os.path.exists(index_file):
        return []
    with open(index_file, "rb") as f:
        index = pickle.load(f)
    return index.get(term, [])

def intersect_documents(postings_lists):
    sets = [set(doc for doc, _ in plist) for plist in postings_lists]
    return set.intersection(*sets)

def compute_tf_idf(postings, common_docs, df, N):
    idf = math.log(N / df)
    scores = {}
    for doc, tf in postings:
        if doc in common_docs:
            scores[doc] = scores.get(doc, 0) + tf * idf
    return scores

def search(query):
    start = time.time()

    terms = tokenize_query(query)
    if not terms:
        print("No valid terms.")
        return []

    postings_lists = [load_postings(term) for term in terms]
    if not all(postings_lists):
        print("No results found.")
        return []

    common_docs = intersect_documents(postings_lists)
    if not common_docs:
        print("No documents contain all terms.")
        return []

    scores = {}
    for term, postings in zip(terms, postings_lists):
        df = len(postings)
        tfidf = compute_tf_idf(postings, common_docs, df, TOTAL_DOCS)
        for doc, score in tfidf.items():
            scores[doc] = scores.get(doc, 0) + score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    end = time.time()
    total_time = round((end - start) * 1000, 2)
    return ranked[:5], total_time

def main():
    queries = [
        "cristina lopes",
        "machine learning",
        "ACM",
        "master of software engineering"
    ]
    for i, q in enumerate(queries, 1):
        print(f"\nQuery {i}: {q}")
        results = search(q)
        for rank, (url, score) in enumerate(results, 1):
            print(f"{rank}. {url} (score: {score:.2f})")

if __name__ == "__main__":
    main()
