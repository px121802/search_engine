# Search Engine from Scratch

A full-text search engine built from scratch in Python that indexes and searches 56,000+ HTML pages from UCI’s domain. The engine supports **boolean queries**, **weighted term ranking**, and a **TF-IDF-based scoring system** with sub-second query response times.

---

## Features

- **Full-Text Indexing**
  - Parses HTML content and extracts title, headings, bold text, and body.
  - Tokenizes and stems words using NLTK's `PorterStemmer`.
  - Assigns weighted term frequencies: title > headings > bold > body.
  - Generates a weighted inverted index with URL references and frequency counts.

- **Efficient Storage**
  - Saves partial inverted indexes to disk to handle large datasets.
  - Merges partial indexes into a single master index.
  - Further splits index by first letter of terms for fast, on-demand loading.

- **Query Processing**
  - Boolean AND queries over multiple terms.
  - TF-IDF ranking of results.
  - Returns top results with relevance scores and query timing.
  - CLI and Flask web interface for user-friendly search.

- **Performance Optimizations**
  - Duplicate HTML detection via MD5 hashing.
  - Partial and bucketed indexes reduce memory footprint.
  - Optimized TF-IDF scoring for sub-300ms query times.

---

## Technologies

- **Backend:** Python, Flask
- **HTML Parsing:** BeautifulSoup
- **Text Processing:** NLTK (tokenization & stemming)
- **Data Storage:** Pickle-based inverted indexes
- **Ranking:** TF-IDF
- **Frontend:** HTML, JavaScript (for Flask interface)
