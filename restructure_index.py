import os
import pickle
import string
from collections import defaultdict

MERGED_INDEX_FILE = "merged_index.pkl"
TERM_INDEX_DIR = "term_index"

def get_bucket(term):
    first = term[0].lower()
    return first if first in string.ascii_lowercase else "other"

def restructure_index():

    with open(MERGED_INDEX_FILE, "rb") as f:
        full_index = pickle.load(f)

    # Split into buckets by first character
    partitioned = defaultdict(lambda: defaultdict(list))

    for term, postings in full_index.items():
        bucket = get_bucket(term)
        partitioned[bucket][term] = postings

    for bucket, partial_index in partitioned.items():
        with open(os.path.join(TERM_INDEX_DIR, f"{bucket}.pkl"), "wb") as f:
            pickle.dump(partial_index, f)

    print(f"Split into {len(partitioned)} files in '{TERM_INDEX_DIR}/'.")

    original_term_count = len(full_index)
    new_term_count = sum(len(pickle.load(open(os.path.join(TERM_INDEX_DIR, f), "rb"))) 
                        for f in os.listdir(TERM_INDEX_DIR) if f.endswith(".pkl"))


if __name__ == "__main__":
    restructure_index()