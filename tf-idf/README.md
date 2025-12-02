# TF-IDF Tag Candidate Extractor

Utility script for extracting TF-IDF–ranked tag candidates from a corpus of security-related articles. The script loads `.txt` documents, applies light preprocessing (lowercasing, URL/HTML cleanup, stop-word removal, phrase normalization), builds a TF-IDF matrix with n-grams, and writes the top terms to CSV for further curation.

## Usage

```bash
python tf-idf/extract.py \
  --data-dir samples \
  --top-k 150 \
  --max-ngram 2 \
  --min-df 2 \
  --max-df 0.6 \
  --output tfidf_top_terms.csv
```

### Arguments
- `--data-dir`: Directory with `.txt` files (default: `samples`).
- `--top-k`: Number of top-scoring terms to include (default: 100).
- `--max-ngram`: Maximum n-gram size; use 2 or 3 to capture short phrases (default: 2).
- `--min-df`: Minimum document frequency to discard extremely rare noise (default: 2).
- `--max-df`: Maximum document frequency proportion to discard overly common words (default: 0.6).
- `--output`: Path to the CSV file that will store `term,score` rows (default: `tfidf_top_terms.csv`).

The generated CSV can be refined manually to merge synonyms, drop IoCs, and map terms to your preferred tag taxonomy.
