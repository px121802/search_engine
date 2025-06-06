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


#This script builds the inverted index from the JSON files in the dataset.
#It parses the HTML, tokenizes, and stems the text and saves the index in multiple chunks before merging them into a final index.




stemmer = PorterStemmer()

#Tokenizes and stems text
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return [stemmer.stem(token.lower()) for token in tokens if token.isalnum()]

#Extracts different parts of HTML
def parse_html(html):
    if not isinstance(html, str):
        return {
            "title": "",
            "headings": "",
            "bold": "",
            "body": ""
        }

    soup = BeautifulSoup(html, "html.parser")

    [s.extract() for s in soup(['script', 'style'])]

    def safe_get_text(tag):
        return tag.get_text() if tag and tag.get_text() else ""

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

#Writes a chunk of the inverted index to disk.
def save_partial_index(index, index_number):
    path = os.path.join(PARTIAL_DIR, f"{index_number}.pkl")
    with open(path, "wb") as f:
        pickle.dump(index, f)

#After processing all documents, this function merges the partial indexes into a single index.
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

    #loops through each document
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
                # Check for duplicate documents using a hash of the HTML content
                doc_hash = md5(html.encode('utf-8')).hexdigest()
                if doc_hash in seen_hashes:
                    continue
                seen_hashes.add(doc_hash)

                text_parts = parse_html(html)
                # Assign weights to term frequencies
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
                #store each term with URL and weighted frequency
                for token, freq in tf.items():
                    inverted_index[token].append((url, freq))
                    unique_tokens.add(token)

                doc_count += 1
                #write to disk every 15000 documents
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
