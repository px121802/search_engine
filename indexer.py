import os
import json
import re
import string
import nltk
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from collections import defaultdict
import pickle

DATA_DIR = "data"
PARTIAL_DIR = "partials"
MERGED_INDEX_FILE = "merged_index.pkl"
DISK_SIZE_LIMIT = 15000

stemmer = PorterStemmer()

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return [stemmer.stem(token.lower()) for token in tokens if token.isalnum()]


def parse_html(html):
    soup = BeautifulSoup(html, "html.parser")
    [s.extract() for s in soup(['script', 'style'])]
    return soup.get_text(separator=' ')

def save_partial_index(index, index_number):
    if not os.path.exists(PARTIAL_DIR):
        os.makedirs(PARTIAL_DIR)
    path = os.path.join(PARTIAL_DIR, f"partial_index_{index_number}.pkl")
    with open(path, "wb") as f:
        pickle.dump(index, f)

def merge_partial_index():
    merged_index = defaultdict(list)
    for filename in os.listdir(PARTIAL_DIR):
        if filename.endswith(".pkl"):
            with open(os.path.join(PARTIAL_DIR, filename), "rb") as f:
                partial = pickle.load(f)
                for term, postings in partial.items():
                    merged_index[term].extend(postings)
    with open(MERGED_INDEX_FILE, "wb") as f:
        pickle.dump(dict(merged_index), f)
    return merged_index


def main():
    doc_count = 0
    unique_tokens = set()
    inverted_index = defaultdict(list)
    partial_index_count = 0

    for domain in os.listdir(DATA_DIR):
        domain_path = os.path.join(DATA_DIR, domain)
        if not os.path.isdir(domain_path):
            continue
        for file in os.listdir(domain_path):
            if not file.endswith(".json"):
                continue
            file_path = os.path.join(domain_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
                url = data.get("url", "")
                html = data.get("content", "")
                text = parse_html(html)
                tokens = tokenize(text)

                tf = defaultdict(int)
                for token in tokens:
                    tf[token] += 1

                for token, freq in tf.items():
                    inverted_index[token].append((url, freq))
                    unique_tokens.add(token)

                doc_count += 1

                if doc_count % DISK_SIZE_LIMIT == 0:
                    save_partial_index(dict(inverted_index), partial_index_count)
                    inverted_index = defaultdict(list)
                    partial_index_count += 1

    if inverted_index:
        save_partial_index(dict(inverted_index), partial_index_count)

    final_index = merge_partial_index()

    size_kb = os.path.getsize(MERGED_INDEX_FILE) / 1024
    print(f"indexed documents: {doc_count}")
    print(f"unique tokens: {len(final_index)}")
    print(f"size of index: {size_kb} KB")

if __name__ == "__main__":
    main()
