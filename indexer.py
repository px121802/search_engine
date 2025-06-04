import os
import json
import re
import string
import nltk
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from collections import defaultdict
import pickle
from hashlib import md5
DATA_DIR = "data"
PARTIAL_DIR = "partial_indexs"
MERGED_INDEX_FILE = "merged_index.pkl"
DISK_SIZE_LIMIT = 15000

stemmer = PorterStemmer()

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return [stemmer.stem(token.lower()) for token in tokens if token.isalnum()]


def parse_html(html):
    # If HTML is missing or not a string, return empty fields
    if not isinstance(html, str):
        return {
            "title": "",
            "headings": "",
            "bold": "",
            "body": ""
        }

    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags
    [s.extract() for s in soup(['script', 'style'])]

    def safe_get_text(tag):
        return tag.get_text() if tag and tag.get_text() else ""

    # Safely extract different parts
    title = soup.title.string if soup.title and soup.title.string else ""
    headings = ' '.join(safe_get_text(tag) for tag in soup.find_all(['h1', 'h2', 'h3']))
    bold = ' '.join(safe_get_text(tag) for tag in soup.find_all(['b', 'strong']))
    body = soup.get_text(separator=' ') or ""

    return {
        "title": title,
        "headings": headings,
        "bold": bold,
        "body": body
    }

def save_partial_index(index, index_number):
    path = os.path.join(PARTIAL_DIR, f"{index_number}.pkl")
    with open(path, "wb") as f:
        pickle.dump(index, f)

def merge_partial_index():
    merged_index = defaultdict(list)
    for filename in os.listdir(PARTIAL_DIR):
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
    seen_hashes = set()

    for domain in os.listdir(DATA_DIR):
        domain_path = os.path.join(DATA_DIR, domain)
        if not os.path.isdir(domain_path):
            continue
        for file in os.listdir(domain_path):
            file_path = os.path.join(domain_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                url = data.get("url", "")
                html = data.get("content", "")

                if not isinstance(html, str) or not html.strip():
                    continue

                doc_hash = md5(html.encode('utf-8')).hexdigest()
                if doc_hash in seen_hashes:
                    continue
                seen_hashes.add(doc_hash)

                text_parts = parse_html(html)

                weights = {
                    "title": 5,
                    "headings": 3,
                    "bold": 2,
                    "body": 1
                }

                tf = defaultdict(int)
                for section, weight in weights.items():
                    tokens = tokenize(text_parts[section])
                    for token in tokens:
                        tf[token] += weight

                for token, freq in tf.items():
                    inverted_index[token].append((url, freq))
                    unique_tokens.add(token)

                doc_count += 1

                if doc_count % DISK_SIZE_LIMIT == 0:
                    save_partial_index(dict(inverted_index), partial_index_count)
                    inverted_index = defaultdict(list)
                    partial_index_count += 1


    save_partial_index(dict(inverted_index), partial_index_count)

    final_index = merge_partial_index()

    size_kb = os.path.getsize(MERGED_INDEX_FILE) / 1024
    print(f"indexed documents: {doc_count}")
    print(f"unique tokens: {len(final_index)}")
    print(f"size of index: {size_kb} KB")

if __name__ == "__main__":
    main()
