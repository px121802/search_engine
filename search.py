import os
import pickle
import math
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import time
import nltk

TERM_INDEX_DIR = "term_index"
TOTAL_DOCS = 55393  # Replace with your final indexed doc count
stemmer = PorterStemmer()

# Tokenizes and stems the query so terms match the index terms.
nltk.download('punkt')

def tokenize_query(query):
    return [stemmer.stem(token.lower()) for token in word_tokenize(query) if token.isalnum()]

#given a term this function returns the path to the index file for that term based on the first letter of the term.
def get_index_file(term):
    first = term[0].lower()
    if first in string.ascii_lowercase:
        return os.path.join(TERM_INDEX_DIR, f"{first}.pkl")
    return os.path.join(TERM_INDEX_DIR, "other.pkl")


# Loads the postings list for a term from its index file.
def load_postings(term):
    index_file = get_index_file(term)
    if not os.path.exists(index_file):
        return []
    with open(index_file, "rb") as f:
        index = pickle.load(f)
    return index.get(term, [])

#implements AND logic by intersection the document sets for all terms in the query
def intersect_documents(postings_lists):
    sets = [set(doc for doc, _ in plist) for plist in postings_lists]
    return set.intersection(*sets)

#computes the TF-IDF score for each document in the postings list.
def compute_tf_idf(postings, common_docs, df, N):
    idf = math.log(N / df)
    scores = {}
    for doc, tf in postings:
        if doc in common_docs:
            scores[doc] = scores.get(doc, 0) + tf * idf
    return scores

def search(query):
    start = time.time()
    #tokenize query terms
    terms = tokenize_query(query)
    if not terms:
        print("No valid terms.")
        return []
    #loads posting for each term from disk
    postings_lists = [load_postings(term) for term in terms]
    if not all(postings_lists):
        print("No results found.")
        return []
    #intersect to find docs that have all terms
    common_docs = intersect_documents(postings_lists)
    if not common_docs:
        print("No documents contain all terms.")
        return []
    #score those docs with TF-IDF
    scores = {}
    for term, postings in zip(terms, postings_lists):
        df = len(postings)
        tfidf = compute_tf_idf(postings, common_docs, df, TOTAL_DOCS)
        for doc, score in tfidf.items():
            scores[doc] = scores.get(doc, 0) + score
    #sort and return top 5 results
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
            print(f"{rank}. {url}")

if __name__ == "__main__":
    main()
