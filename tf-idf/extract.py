"""
TF-IDF tag candidate extractor.

Loads text files from a directory, performs light preprocessing, builds a TF-IDF matrix
(with n-grams), and writes the top-N scored terms to CSV. Designed for bilingual
(Russian/English) corpora common in security research writeups.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer


PHRASE_REPLACEMENTS = {
    "active directory": "active_directory",
    "windows defender": "windows_defender",
    "cobalt strike": "cobalt_strike",
    "mimikatz": "mimikatz",
    "ntlm relay": "ntlm_relay",
    "privilege escalation": "privilege_escalation",
    "edr bypass": "edr_bypass",
    "driver vulnerability": "driver_vulnerability",
}


CUSTOM_STOPWORDS = {
    # Domain-generic Russian
    "система",
    "данный",
    "используется",
    "позволяет",
    "пользователь",
    "данных",
    "также",
    "время",
    "однако",
    "может",
    "который",
    "которые",
    "таким",
    "образом",
    "решение",
    "получить",
    "пример",
    "согласно",
    "статья",
    "сети",
    "атакующий",
    # English filler
    "system",
    "user",
    "data",
    "allows",
    "used",
    "example",
    "however",
    "solution",
    "article",
    # Organization placeholders
    "company",
}


IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
URL_RE = re.compile(r"https?://\S+")
HASH_RE = re.compile(r"\b[a-f0-9]{32,64}\b")


def load_texts(text_dir: Path) -> List[str]:
    texts: List[str] = []
    for path in sorted(text_dir.glob("*.txt")):
        content = path.read_text(encoding="utf-8", errors="ignore")
        texts.append(content)
    if not texts:
        raise ValueError(f"No .txt files found in {text_dir}")
    return texts


def apply_phrase_replacements(text: str, replacements: dict[str, str]) -> str:
    processed = text
    for phrase, replacement in replacements.items():
        processed = re.sub(re.escape(phrase.lower()), replacement, processed, flags=re.IGNORECASE)
    return processed


def preprocess_text(raw: str, stopwords: Sequence[str]) -> str:
    lowercased = raw.lower()
    without_urls = URL_RE.sub(" ", lowercased)
    without_html = re.sub(r"<[^>]+>", " ", without_urls)
    with_phrases = apply_phrase_replacements(without_html, PHRASE_REPLACEMENTS)
    cleaned = re.sub(r"[^a-zа-я0-9_\-/]+", " ", with_phrases)
    tokens = cleaned.split()
    filtered_tokens: List[str] = []
    for token in tokens:
        if token in stopwords:
            continue
        if len(token) < 3:
            continue
        filtered_tokens.append(token)
    return " ".join(filtered_tokens)


def build_stopwords() -> set[str]:
    stopwords = set(text.ENGLISH_STOP_WORDS)
    stopwords.update(CUSTOM_STOPWORDS)
    stopwords.update(
        {
            "и",
            "в",
            "во",
            "не",
            "что",
            "он",
            "она",
            "они",
            "как",
            "а",
            "но",
            "к",
            "до",
            "после",
            "при",
            "это",
            "этот",
            "для",
            "или",
            "про",
            "над",
            "под",
            "из",
            "о",
            "на",
            "с",
            "быть",
            "есть",
            "так",
            "же",
            "мы",
            "вы",
            "их",
            "ему",
            "её",
            "них",
        }
    )
    return stopwords


def term_is_noise(term: str) -> bool:
    if len(term) < 3:
        return True
    if IPV4_RE.search(term):
        return True
    if URL_RE.search(term):
        return True
    if HASH_RE.search(term):
        return True
    alnum = re.sub(r"[^a-zа-я0-9]", "", term)
    if not alnum:
        return True
    digit_ratio = sum(ch.isdigit() for ch in term) / len(term)
    if digit_ratio >= 0.5:
        return True
    return False


def compute_tfidf_scores(texts: Iterable[str], stopwords: Sequence[str], max_ngram: int, min_df: int, max_df: float) -> tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, max_ngram),
        min_df=min_df,
        max_df=max_df,
        stop_words=stopwords,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    return scores, terms


def filter_and_sort_terms(scores: np.ndarray, terms: np.ndarray, top_k: int) -> List[tuple[str, float]]:
    scored_terms = [
        (term, float(score)) for term, score in zip(terms, scores) if not term_is_noise(term)
    ]
    scored_terms.sort(key=lambda item: item[1], reverse=True)
    return scored_terms[:top_k]


def write_to_csv(rows: Sequence[tuple[str, float]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["term", "score"])
        for term, score in rows:
            writer.writerow([term, f"{score:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract TF-IDF tag candidates from text files.")
    parser.add_argument("--data-dir", type=Path, default=Path("samples"), help="Directory with .txt files.")
    parser.add_argument("--top-k", type=int, default=100, help="Number of top terms to output.")
    parser.add_argument("--max-ngram", type=int, default=2, help="Maximum n-gram size (1-3 recommended).")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency.")
    parser.add_argument("--max-df", type=float, default=0.6, help="Maximum document frequency proportion.")
    parser.add_argument("--output", type=Path, default=Path("tfidf_top_terms.csv"), help="Output CSV path.")

    args = parser.parse_args()

    stopwords = sorted(build_stopwords())
    texts_raw = load_texts(args.data_dir)
    preprocessed = [preprocess_text(text, stopwords) for text in texts_raw]
    scores, terms = compute_tfidf_scores(
        preprocessed,
        stopwords,
        max_ngram=args.max_ngram,
        min_df=args.min_df,
        max_df=args.max_df,
    )
    top_terms = filter_and_sort_terms(scores, terms, top_k=args.top_k)
    write_to_csv(top_terms, args.output)

    print(f"Processed {len(preprocessed)} documents from {args.data_dir}.")
    print(f"Top {len(top_terms)} terms written to {args.output}.")
    for term, score in top_terms[:20]:
        print(f"{term:40s} {score:.4f}")


if __name__ == "__main__":
    main()
