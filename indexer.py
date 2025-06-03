# indexer.py - Builds the inverted index for the search engine
# ICS 121 - Information Retrieval
# This file processes the ICS website data and creates an inverted index for fast searching

import os
import json
import re
import string
import nltk
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)  # ignore BS4 warnings
from nltk.stem import PorterStemmer
from collections import defaultdict
import pickle
import math
from typing import Dict, List, Set, Tuple
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil
import gc

# Where to find stuff
DATA_DIR = "data/DEV"  # folder with the ICS website data
INDEX_DIR = "index"  # where we'll save our index
PARTIAL_INDEX_DIR = "partial_indexes"  # temp storage while indexing
DOC_META_FILE = "doc_meta.pkl"  # stores document metadata
TERMS_META_FILE = "terms_meta.pkl"  # stores term metadata
POSITIONS_DIR = "positions"
SEGMENT_SIZE = 50000  # terms per index segment
OFFLOAD_THRESHOLD = 15000  # docs to process before saving
MAX_WORKERS = 4  # parallel threads
MEMORY_THRESHOLD = 80  # when to write to disk (memory %)

stemmer = PorterStemmer()  # for word stemming
process_lock = threading.Lock()  # thread safety

class IndexEntry:
    """Stores posting list info efficiently"""
    __slots__ = ['postings', 'df']  # using slots to save memory
    
    def __init__(self):
        self.postings = []  # list of (url, weight, positions)
        self.df = 0  # number of docs containing term
    
    def add_posting(self, url: str, weight: float, positions: List[int]):
        self.postings.append((url, weight, positions))
        self.df += 1
    
    def merge(self, other: 'IndexEntry'):
        """Combines two posting lists"""
        self.postings.extend(other.postings)
        self.df += other.df

def check_memory_usage() -> bool:
    """Returns True if we're using too much memory"""
    memory = psutil.virtual_memory()
    return memory.percent >= MEMORY_THRESHOLD

def save_partial_index(terms: Dict[str, IndexEntry], partial_num: int):
    """Saves part of the index to disk when memory gets full"""
    if not os.path.exists(PARTIAL_INDEX_DIR):
        os.makedirs(PARTIAL_INDEX_DIR)
    
    filename = os.path.join(PARTIAL_INDEX_DIR, f"partial_{partial_num}.npz")
    
    # Group data by term for numpy arrays
    term_data = defaultdict(lambda: {'urls': [], 'weights': [], 'positions': []})
    
    for term, entry in terms.items():
        for url, weight, positions in entry.postings:
            term_data[term]['urls'].append(url)
            term_data[term]['weights'].append(weight)
            term_data[term]['positions'].append(positions)
    
    # Save as compressed numpy arrays to save space
    data_arrays = {}
    for term, data in term_data.items():
        data_arrays[f"{term}_urls"] = np.array(data['urls'], dtype=object)
        data_arrays[f"{term}_weights"] = np.array(data['weights'], dtype=np.float32)
        data_arrays[f"{term}_positions"] = np.array(data['positions'], dtype=object)
    
    np.savez_compressed(filename, **data_arrays)
    print(f"Saved partial index {partial_num}")

def merge_partial_indexes():
    """Combines all the partial indexes into final segments"""
    if not os.path.exists(PARTIAL_INDEX_DIR):
        return
    
    print("\nMerging partial indexes...")
    merged_terms = defaultdict(IndexEntry)
    partial_files = sorted(os.listdir(PARTIAL_INDEX_DIR))
    
    for partial_file in tqdm(partial_files):
        if not partial_file.endswith('.npz'):
            continue
        
        filepath = os.path.join(PARTIAL_INDEX_DIR, partial_file)
        with np.load(filepath, allow_pickle=True) as data:
            # Process terms in chunks to manage memory
            terms = set(k.split('_')[0] for k in data.files)
            for term in terms:
                if f"{term}_urls" in data:
                    urls = data[f"{term}_urls"]
                    weights = data[f"{term}_weights"]
                    positions = data[f"{term}_positions"]
                    
                    for url, weight, pos in zip(urls, weights, positions):
                        merged_terms[term].add_posting(url, float(weight), pos)
                
                # Save segment if memory gets full
                if len(merged_terms) >= SEGMENT_SIZE or check_memory_usage():
                    save_segment(merged_terms, len(os.listdir(INDEX_DIR)))
                    merged_terms.clear()
                    gc.collect()
    
    # Save any remaining terms
    if merged_terms:
        save_segment(merged_terms, len(os.listdir(INDEX_DIR)))
    
    # Clean up temp files
    for file in partial_files:
        os.remove(os.path.join(PARTIAL_INDEX_DIR, file))
    os.rmdir(PARTIAL_INDEX_DIR)

def tokenize_with_positions(text: str, offset: int = 0) -> List[Tuple[str, int]]:
    """Splits text into tokens and tracks their positions"""
    tokens = nltk.word_tokenize(text.lower())
    return [(stemmer.stem(token), pos + offset) for pos, token in enumerate(tokens) if token.isalnum()]

def parse_html(html: str) -> Dict:
    """Extracts text from HTML and assigns weights to different sections"""
    try:
        soup = BeautifulSoup(html, "lxml")
    except:
        # Fall back to slower parser if lxml fails
        soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style tags
    [s.extract() for s in soup(['script', 'style'])]
    
    current_pos = 0
    section_tokens = {}
    
    # Weight different HTML sections (tuned these weights by testing)
    sections = {
        'title': {'weight': 5.0, 'tags': ['title']},  # titles are most important
        'h1': {'weight': 4.0, 'tags': ['h1']},
        'h2': {'weight': 3.0, 'tags': ['h2']},
        'h3': {'weight': 2.0, 'tags': ['h3']},
        'bold': {'weight': 1.5, 'tags': ['b', 'strong']},
        'text': {'weight': 1.0, 'tags': None}  # regular text
    }
    
    for section, config in sections.items():
        if config['tags']:
            text = ' '.join(t.get_text() for t in soup.find_all(config['tags']))
        else:
            text = soup.get_text(separator=' ')
        
        tokens_with_pos = tokenize_with_positions(text, current_pos)
        section_tokens[section] = {
            'tokens': tokens_with_pos,
            'weight': config['weight']
        }
        current_pos += len(tokens_with_pos)
    
    return section_tokens

def process_document(url: str, html: str) -> Tuple[Dict[str, IndexEntry], Dict]:
    """Processes a single webpage and extracts its terms"""
    section_tokens = parse_html(html)
    
    # Track term info
    term_info = defaultdict(lambda: {'positions': [], 'weight': 0.0})
    doc_length = 0
    
    for section, data in section_tokens.items():
        weight = data['weight']
        for token, pos in data['tokens']:
            term_info[token]['positions'].append(pos)
            term_info[token]['weight'] += weight
            doc_length += 1
    
    # Create index entries for each term
    doc_terms = {}
    for term, info in term_info.items():
        entry = IndexEntry()
        entry.add_posting(url, info['weight'], info['positions'])
        doc_terms[term] = entry
    
    # Save document stats
    doc_meta = {
        'length': doc_length,
        'max_tf': max(len(info['positions']) for info in term_info.values()) if term_info else 0
    }
    
    return doc_terms, doc_meta

def save_segment(terms: Dict[str, IndexEntry], segment_num: int):
    """Saves a chunk of the index to disk"""
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    
    # Group by term for numpy arrays
    term_data = defaultdict(lambda: {'urls': [], 'weights': [], 'positions': []})
    
    for term, entry in terms.items():
        for url, weight, positions in entry.postings:
            term_data[term]['urls'].append(url)
            term_data[term]['weights'].append(weight)
            term_data[term]['positions'].append(positions)
    
    # Save as compressed numpy arrays
    filename = os.path.join(INDEX_DIR, f"segment_{segment_num}.npz")
    data_arrays = {}
    
    for term, data in term_data.items():
        data_arrays[f"{term}_urls"] = np.array(data['urls'], dtype=object)
        data_arrays[f"{term}_weights"] = np.array(data['weights'], dtype=np.float32)
        data_arrays[f"{term}_positions"] = np.array(data['positions'], dtype=object)
    
    np.savez_compressed(filename, **data_arrays)

def process_file_batch(file_batch: List[str]) -> Tuple[Dict[str, IndexEntry], Dict[str, Dict]]:
    """Processes a batch of files in parallel"""
    batch_terms = defaultdict(IndexEntry)
    batch_meta = {}
    
    for file_path in file_batch:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                url = data.get("url", "")
                html = data.get("content", "")
                
                # Process the webpage
                doc_terms, meta = process_document(url, html)
                
                # Thread-safe update of batch data
                with process_lock:
                    batch_meta[url] = meta
                    for term, entry in doc_terms.items():
                        batch_terms[term].postings.extend(entry.postings)
                        batch_terms[term].df += entry.df
                
        except json.JSONDecodeError:
            continue
    
    return batch_terms, batch_meta

def main():
    """Main indexing process"""
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    
    print("Phase 1: Finding all the webpages...")
    all_files = []
    for domain in os.listdir(DATA_DIR):
        domain_path = os.path.join(DATA_DIR, domain)
        if os.path.isdir(domain_path):
            for file in os.listdir(domain_path):
                if file.endswith(".json"):
                    all_files.append(os.path.join(domain_path, file))
    
    doc_count = len(all_files)
    print(f"Found {doc_count} webpages")
    
    # Process files in parallel
    doc_meta = {}
    all_terms = defaultdict(IndexEntry)
    current_segment = 0
    segment_terms = defaultdict(IndexEntry)
    partial_index_count = 0
    
    # Split into batches for parallel processing
    batch_size = 100  # found this to be a good size
    file_batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]
    
    print("\nPhase 2: Processing webpages...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for batch in file_batches:
            future = executor.submit(process_file_batch, batch)
            futures.append(future)
        
        # Process results as they come in
        docs_processed = 0
        for future in tqdm(futures, total=len(file_batches)):
            batch_terms, batch_meta = future.result()
            
            # Update our data
            doc_meta.update(batch_meta)
            
            for term, entry in batch_terms.items():
                segment_terms[term].postings.extend(entry.postings)
                segment_terms[term].df += entry.df
                all_terms[term].df += entry.df
            
            docs_processed += len(batch_meta)
            
            # Save to disk if memory gets full
            if docs_processed >= OFFLOAD_THRESHOLD or check_memory_usage():
                print(f"\nSaving partial index {partial_index_count}...")
                save_partial_index(segment_terms, partial_index_count)
                partial_index_count += 1
                segment_terms.clear()
                gc.collect()
    
    # Combine all the partial indexes
    merge_partial_indexes()
    
    # Save term info
    terms_meta = {term: {'df': entry.df, 'segment': i // SEGMENT_SIZE} 
                 for i, (term, entry) in enumerate(sorted(all_terms.items()))}
    
    # Save metadata files
    with open(DOC_META_FILE, "wb") as f:
        pickle.dump(doc_meta, f)
    with open(TERMS_META_FILE, "wb") as f:
        pickle.dump(terms_meta, f)
    
    print(f"\nAll done!")
    print(f"Total webpages indexed: {len(doc_meta)}")
    print(f"Total unique terms: {len(terms_meta)}")
    print(f"Partial indexes created: {partial_index_count}")
    print(f"Final index segments: {len(os.listdir(INDEX_DIR))}")

if __name__ == "__main__":
    main()
