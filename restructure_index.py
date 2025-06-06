import os
import pickle
import string
from collections import defaultdict
#this script takes the merged index and restructures it into smaller files based on the first letter of each term
MERGED_INDEX_FILE = "merged_index.pkl"
TERM_INDEX_DIR = "term_index"

#This function decides which file a term belongs to based on the first letter of the term
def get_bucket(term):
    first = term[0].lower()
    return first if first in string.ascii_lowercase else "other"


#This function loads the full inverted index and loops through every term and adding it to the correct bucket based on the first letter of the term.
def restructure_index():

    with open(MERGED_INDEX_FILE, "rb") as f:
        full_index = pickle.load(f)
    partitioned = defaultdict(lambda: defaultdict(list))

    for term, postings in full_index.items():
        bucket = get_bucket(term)
        partitioned[bucket][term] = postings

    for bucket, partial_index in partitioned.items():
        with open(os.path.join(TERM_INDEX_DIR, f"{bucket}.pkl"), "wb") as f:
            pickle.dump(partial_index, f)

    print(f"split done")

if __name__ == "__main__":
    restructure_index()