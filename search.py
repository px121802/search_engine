# search.py - Handles searching through the inverted index
# ICS 121 - Information Retrieval
# This file implements the search functionality for the ICS website search engine

import os
import pickle
import math
import nltk
import mmap
from nltk.stem import PorterStemmer
from collections import defaultdict, OrderedDict
from typing import List, Dict, Set, Tuple, Optional
import time
from functools import lru_cache
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import heapq
import psutil
from multiprocessing import Pool, cpu_count
import threading
import json
from indexer import tokenize_with_positions

# File paths and configuration
INDEX_DIR = "index"  # where we store our index segments
DOC_META_FILE = "doc_meta.pkl"  # stores document info like length
TERMS_META_FILE = "terms_meta.pkl"  # stores term info like document frequency

# Performance tuning parameters
MAX_CACHE_SEGMENTS = 5  # how many segments to keep in memory
MAX_WORKERS = min(4, cpu_count())  # number of parallel workers
BATCH_SIZE = 200  # docs to process at once
MIN_MEMORY_PCT = 20  # when to start clearing memory
MAX_RESULTS = 10  # max search results to return
CACHE_SIZE = 1000  # size of term cache

stemmer = PorterStemmer()  # for word stemming (e.g. running -> run)

# Load the metadata we need for searching
with open(DOC_META_FILE, "rb") as f:
    doc_meta = pickle.load(f)  # contains doc lengths etc

with open(TERMS_META_FILE, "rb") as f:
    terms_meta = pickle.load(f)  # contains term frequencies etc

# Calculate some stats we need for BM25 scoring
total_docs = len(doc_meta)
avg_doc_length = sum(meta['length'] for meta in doc_meta.values()) / total_docs if total_docs > 0 else 0

# BM25 parameters (tuned these by testing different values)
k1 = 1.2  # term frequency saturation parameter
b = 0.75  # length normalization parameter

# Thread stuff for parallel processing
thread_local = threading.local()

class MMAPCache:
    """Caches memory-mapped index files to avoid reopening them"""
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.cache = {}
        self.lock = threading.Lock()
    
    def get(self, segment_num: int) -> Optional[mmap.mmap]:
        with self.lock:
            if segment_num in self.cache:
                return self.cache[segment_num]
            
            # Remove oldest if cache is full
            if len(self.cache) >= self.max_size:
                oldest = next(iter(self.cache))
                self.cache[oldest].close()
                del self.cache[oldest]
            
            try:
                filepath = os.path.join(INDEX_DIR, f"segment_{segment_num}.npz")
                with open(filepath, "rb") as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    self.cache[segment_num] = mm
                    return mm
            except:
                return None

mmap_cache = MMAPCache()

@lru_cache(maxsize=CACHE_SIZE)
def get_term_postings(term: str, segment_num: int) -> Tuple[List[str], List[float], List[List[int]]]:
    """Gets the postings list for a term from its segment file"""
    try:
        filepath = os.path.join(INDEX_DIR, f"segment_{segment_num}.npz")
        with np.load(filepath, allow_pickle=True) as data:
            # Each term has 3 arrays: urls, weights, and positions
            if f"{term}_urls" not in data:
                return [], [], []
            
            urls = data[f"{term}_urls"].tolist()
            weights = data[f"{term}_weights"].tolist()
            positions = data[f"{term}_positions"].tolist()
            
            return urls, weights, positions
    except Exception as e:
        print(f"Error loading segment {segment_num}: {str(e)}")
        return [], [], []

def calculate_bm25_score(url: str, query_terms: List[str], k1: float = 1.5, b: float = 0.75) -> float:
    """Calculates BM25 relevance score for a document"""
    score = 0
    doc_length = doc_meta[url]['length']
    # Normalize for document length
    length_norm = (1 - b) + b * (doc_length / avg_doc_length)
    
    for term in query_terms:
        if term not in terms_meta:
            continue
            
        segment_num = terms_meta[term]['segment']
        urls, _, positions = get_term_postings(term, segment_num)
        
        if url in urls:
            idx = urls.index(url)
            tf = len(positions[idx])  # term frequency
            df = terms_meta[term]['df']  # document frequency
            # IDF calculation
            idf = math.log((total_docs - df + 0.5) / (df + 0.5))
            
            # BM25 formula
            tf_component = ((k1 + 1) * tf) / (k1 * length_norm + tf)
            
            score += idf * tf_component
    
    return score

def process_phrase_positions(positions1: List[int], positions2: List[int], distance: int = 1) -> float:
    """Calculate phrase matching score based on term positions"""
    if not positions1 or not positions2:
        return 0.0
    
    score = 0.0
    for pos1 in positions1:
        for pos2 in positions2:
            if abs(pos2 - pos1) == distance:
                score += 1.0
    
    return score

def search_segment(term: str, segment_num: int) -> Dict[str, Tuple[float, List[int]]]:
    """Search for a term in a specific segment"""
    urls, weights, positions = get_term_postings(term, segment_num)
    
    results = {}
    for url, weight, pos in zip(urls, weights, positions):
        doc_length = doc_meta[url]['length']
        tf = len(pos)
        df = terms_meta[term]['df']
        
        score = calculate_bm25_score(url, [term]) * weight
        results[url] = (score, pos)
    
    return results

def merge_results(results_list: List[Dict[str, Tuple[float, List[int]]]]) -> Dict[str, Tuple[float, Dict[str, List[int]]]]:
    """Merge results from multiple segments"""
    merged = defaultdict(lambda: (0.0, {}))
    
    for results in results_list:
        for url, (score, positions) in results.items():
            current_score, current_positions = merged[url]
            merged[url] = (current_score + score, {**current_positions, **{'positions': positions}})
    
    return dict(merged)

def search(query: str) -> List[Dict]:
    """
    Main search function - takes a query string and returns ranked results
    Uses BM25 for scoring and supports OR queries (any term can match)
    """
    # Get stemmed query terms
    query_terms = [term for term, _ in tokenize_with_positions(query)]
    
    if not query_terms:
        return []
    
    # Debug info about terms
    print("Term existence in index:")
    for term in query_terms:
        if term in terms_meta:
            print(f"- {term}: Found (df={terms_meta[term]['df']})")
        else:
            print(f"- {term}: Not found")
    
    # Find all docs containing any query term
    matching_docs = set()
    for term in query_terms:
        if term not in terms_meta:
            continue
        
        segment_num = terms_meta[term]['segment']
        urls, _, _ = get_term_postings(term, segment_num)
        matching_docs.update(urls)
        print(f"\nDocuments containing '{term}': {len(urls)}")
        if len(urls) < 5:
            print(f"URLs: {urls}")
        else:
            print(f"Sample URLs: {urls[:5]}")
    
    print(f"\nTotal documents matching any term: {len(matching_docs)}")
    if len(matching_docs) < 5:
        print(f"Matching URLs: {list(matching_docs)}")
    else:
        print(f"Sample matching URLs: {list(matching_docs)[:5]}")
    
    # Score each matching document
    results = []
    for url in matching_docs:
        score = calculate_bm25_score(url, query_terms)
        results.append({
            'url': url,
            'score': score
        })
    
    # Sort by score and return top 5
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:5]

def main():
    """Test the search engine with the required queries"""
    # Debug: Print index stats
    print(f"Total documents in index: {len(doc_meta)}")
    print(f"Total terms in index: {len(terms_meta)}")
    print("\nSample terms from index:")
    sample_terms = list(terms_meta.keys())[:5]
    for term in sample_terms:
        print(f"- {term}: {terms_meta[term]}")
    
    test_queries = [
        "cristina lopes",
        "machine learning",
        "ACM",
        "master of software engineering"
    ]
    
    print("\nTesting milestone 2 queries...")
    for query in test_queries:
        print(f"\nQuery: {query}")
        # Debug: Print tokenized terms
        query_terms = [term for term, _ in tokenize_with_positions(query)]
        print(f"Tokenized terms: {query_terms}")
        
        results = search(query)
        print("Top 5 results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['url']}")

if __name__ == "__main__":
    main() 