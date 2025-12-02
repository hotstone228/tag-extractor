import argparse
import csv
import html
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

try:  # Optional: only used when installed
    import pymorphy2
except ImportError:  # pragma: no cover - optional dependency
    pymorphy2 = None


RUSSIAN_STOP_WORDS: set[str] = {
    "и",
    "в",
    "во",
    "не",
    "что",
    "он",
    "на",
    "я",
    "с",
    "со",
    "как",
    "а",
    "то",
    "все",
    "она",
    "так",
    "его",
    "но",
    "да",
    "ты",
    "к",
    "у",
    "же",
    "вы",
    "за",
    "бы",
    "по",
    "ее",
    "мне",
    "есть",
    "они",
    "тут",
    "где",
    "при",
    "система",
    "данный",
    "позволяет",
    "используется",
}

DOMAIN_STOP_WORDS: set[str] = {
    "атака",
    "система",
    "пользователь",
    "данные",
    "данный",
    "использование",
    "используется",
    "аналитика",
}


def load_documents(input_dir: Path) -> list[str]:
    documents: list[str] = []
    for path in sorted(input_dir.rglob("*.txt")):
        documents.append(path.read_text(encoding="utf-8", errors="ignore"))
    if not documents:
        raise ValueError(f"No .txt files found in {input_dir}")
    return documents


def normalize_entities(text: str, replacements: dict[str, str]) -> str:
    for original, normalized in replacements.items():
        text = re.sub(original, normalized, text, flags=re.IGNORECASE)
    return text


def clean_text(text: str, replacements: dict[str, str]) -> str:
    text = html.unescape(text)
    text = normalize_entities(text, replacements)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"[`*_#>\\-]+", " ", text)
    text = re.sub(r"[^a-zа-я0-9_\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lemmatize_tokens(tokens: Iterable[str]) -> List[str]:
    if not pymorphy2:
        return list(tokens)
    analyzer = pymorphy2.MorphAnalyzer()
    return [analyzer.parse(token)[0].normal_form for token in tokens]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zа-я0-9_\-]+", text)


def build_stopwords(extra_stopwords: Sequence[str]) -> set[str]:
    stopwords = set(ENGLISH_STOP_WORDS)
    stopwords.update(RUSSIAN_STOP_WORDS)
    stopwords.update(DOMAIN_STOP_WORDS)
    stopwords.update({word.lower() for word in extra_stopwords})
    stopwords = {word.replace(" ", "_") for word in stopwords}
    return stopwords


def prepare_corpus(documents: list[str], replacements: dict[str, str], lemmatize: bool,
                   extra_stopwords: Sequence[str]) -> tuple[list[str], set[str]]:
    stopwords = build_stopwords(extra_stopwords)
    processed_docs: list[str] = []
    for doc in documents:
        cleaned = clean_text(doc, replacements)
        tokens = tokenize(cleaned)
        tokens = [token for token in tokens if token not in stopwords]
        if lemmatize:
            tokens = lemmatize_tokens(tokens)
        tokens = [token for token in tokens if token not in stopwords]
        processed_docs.append(" ".join(tokens))
    return processed_docs, stopwords


def compute_tfidf_scores(texts: list[str], stopwords: set[str],
                         ngram_range: tuple[int, int], min_df: int,
                         max_df: float) -> tuple[np.ndarray, list[str]]:
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words=stopwords,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out().tolist()
    scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    return scores, terms


def is_noise(term: str) -> bool:
    if len(term) < 3:
        return True
    if re.fullmatch(r"[0-9\-_.]+", term):
        return True
    if re.search(r"(https?://|[0-9]{1,3}(\.[0-9]{1,3}){3})", term):
        return True
    hex_ratio = sum(c in "0123456789abcdef" for c in term.lower()) / max(len(term), 1)
    if hex_ratio > 0.7 and len(term) > 6:
        return True
    return False


def rank_terms(scores: np.ndarray, terms: list[str], top_n: int) -> list[tuple[str, float]]:
    sorted_idx = np.argsort(scores)[::-1]
    ranked: list[tuple[str, float]] = []
    for idx in sorted_idx:
        term = terms[idx]
        score = float(scores[idx])
        if is_noise(term):
            continue
        ranked.append((term, score))
        if len(ranked) >= top_n:
            break
    return ranked


def write_to_csv(rows: list[tuple[str, float]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["term", "score"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract candidate tags using TF-IDF")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory with .txt documents")
    parser.add_argument("--output", type=Path, default=Path("candidates.csv"),
                        help="Path to the output CSV file")
    parser.add_argument("--top-n", type=int, default=500,
                        help="How many top terms to keep")
    parser.add_argument("--min-df", type=int, default=3,
                        help="Minimum documents for a term")
    parser.add_argument("--max-df", type=float, default=0.5,
                        help="Maximum document frequency")
    parser.add_argument("--ngram-max", type=int, default=2,
                        help="Maximum n-gram length (min is 1)")
    parser.add_argument("--lemmatize", action="store_true",
                        help="Enable lemmatization (requires pymorphy2)")
    parser.add_argument("--stopword", action="append", default=[],
                        help="Extra stopwords to exclude")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replacements = {
        r"active\s+directory": "Active_Directory",
        r"windows\s+defender": "Windows_Defender",
        r"cobalt\s+strike": "Cobalt_Strike",
        r"mimikatz": "Mimikatz",
        r"ntlm\s+relay": "NTLM_Relay",
    }

    documents = load_documents(args.input_dir)
    texts, stopwords = prepare_corpus(
        documents,
        replacements=replacements,
        lemmatize=args.lemmatize,
        extra_stopwords=args.stopword,
    )

    scores, terms = compute_tfidf_scores(
        texts,
        stopwords=stopwords,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
    )

    ranked_terms = rank_terms(scores, terms, top_n=args.top_n)
    write_to_csv(ranked_terms, args.output)


if __name__ == "__main__":
    main()
