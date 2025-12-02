from __future__ import annotations

import argparse
import csv
import importlib.util
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional lemmatizers discovered dynamically to avoid hard failures when they
# are not installed. Imports are only attempted when the packages are present.
pymorphy2_spec = importlib.util.find_spec("pymorphy2")
if pymorphy2_spec:
    import pymorphy2  # type: ignore
else:
    pymorphy2 = None


def build_default_replacements() -> dict[str, str]:
    return {
        "active directory": "active_directory",
        "windows defender": "windows_defender",
        "cobalt strike": "cobalt_strike",
        "mimikatz": "mimikatz",
    }


def load_documents(source: Path) -> List[str]:
    if source.is_file():
        return [source.read_text(encoding="utf-8", errors="ignore")]

    documents: List[str] = []
    for path in sorted(source.rglob("*.txt")):
        documents.append(path.read_text(encoding="utf-8", errors="ignore"))
    if not documents:
        raise ValueError(f"No .txt documents found under {source}")
    return documents


EN_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

RU_STOPWORDS = {
    "а",
    "без",
    "более",
    "бы",
    "был",
    "была",
    "были",
    "было",
    "быть",
    "в",
    "вам",
    "вами",
    "вас",
    "весь",
    "во",
    "вот",
    "все",
    "всего",
    "всех",
    "всю",
    "вы",
    "где",
    "да",
    "даже",
    "для",
    "до",
    "его",
    "ее",
    "ей",
    "ему",
    "если",
    "есть",
    "еще",
    "же",
    "за",
    "здесь",
    "из",
    "или",
    "им",
    "ими",
    "их",
    "к",
    "как",
    "какая",
    "какой",
    "кто",
    "ли",
    "либо",
    "мне",
    "много",
    "может",
    "мы",
    "на",
    "над",
    "надо",
    "наш",
    "не",
    "него",
    "нее",
    "ней",
    "нет",
    "ни",
    "ниже",
    "них",
    "но",
    "ну",
    "о",
    "об",
    "однако",
    "он",
    "она",
    "они",
    "оно",
    "от",
    "очень",
    "по",
    "под",
    "после",
    "потому",
    "при",
    "про",
    "раз",
    "с",
    "со",
    "так",
    "также",
    "такой",
    "там",
    "тебя",
    "тем",
    "теми",
    "теперь",
    "то",
    "тобой",
    "тобою",
    "тоже",
    "только",
    "тот",
    "тою",
    "три",
    "тут",
    "ты",
    "у",
    "уж",
    "уже",
    "хотя",
    "чего",
    "чей",
    "чем",
    "что",
    "чтобы",
    "чье",
    "чья",
    "эта",
    "эти",
    "это",
    "я",
}


def merge_stopwords(extra_stopwords: Iterable[str]) -> set[str]:
    combined = set(EN_STOPWORDS) | set(RU_STOPWORDS)
    combined |= {word.strip().lower() for word in extra_stopwords if word.strip()}
    return combined


def compile_replacements(raw_replacements: dict[str, str]) -> list[tuple[re.Pattern[str], str]]:
    compiled: list[tuple[re.Pattern[str], str]] = []
    for phrase, token in raw_replacements.items():
        pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
        compiled.append((pattern, token))
    return compiled


def strip_markup(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[(.*?)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zа-я0-9_][\w-]*", text, flags=re.IGNORECASE)


def normalize_tokens(tokens: Sequence[str], stopwords: set[str]) -> list[str]:
    morph = pymorphy2.MorphAnalyzer() if pymorphy2 else None
    normalized: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in stopwords:
            continue
        lemma = lowered
        if morph and re.search(r"[а-яё]", lowered):
            parse = morph.parse(lowered)
            if parse:
                lemma = parse[0].normal_form
        if lemma in stopwords:
            continue
        normalized.append(lemma)
    return normalized


def preprocess(text: str, replacements: list[tuple[re.Pattern[str], str]], stopwords: set[str]) -> str:
    text = text.lower()
    for pattern, token in replacements:
        text = pattern.sub(token, text)
    text = strip_markup(text)
    text = re.sub(r"[^\w\s-]", " ", text)
    tokens = tokenize(text)
    normalized_tokens = normalize_tokens(tokens, stopwords)
    return " ".join(normalized_tokens)


def build_vectorizer(args: argparse.Namespace) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, args.max_ngram),
        min_df=args.min_df,
        max_df=args.max_df,
        token_pattern=r"(?u)\b\w[\w-]*\b",
        lowercase=False,
    )


def compute_scores(matrix, scoring: str) -> np.ndarray:
    if scoring == "mean":
        scores = np.asarray(matrix.mean(axis=0)).ravel()
    elif scoring == "max":
        scores = np.asarray(matrix.max(axis=0)).ravel()
    elif scoring == "sum":
        scores = np.asarray(matrix.sum(axis=0)).ravel()
    else:
        raise ValueError(f"Unsupported scoring method: {scoring}")
    return scores


def is_ioc_like(term: str) -> bool:
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", term):
        return True
    if re.fullmatch(r"[0-9a-f]{32,64}", term):
        return True
    if re.match(r"^[\w.-]+\.[a-z]{2,}$", term):
        return True
    return False


def filter_terms(terms: Sequence[str], scores: Sequence[float], min_length: int) -> list[tuple[str, float]]:
    filtered: list[tuple[str, float]] = []
    for term, score in zip(terms, scores):
        if len(term) < min_length:
            continue
        digit_ratio = sum(char.isdigit() for char in term) / len(term)
        if digit_ratio > 0.5:
            continue
        if is_ioc_like(term):
            continue
        filtered.append((term, float(score)))
    return filtered


def save_to_csv(rows: Sequence[tuple[str, float]], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "score"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract candidate tags using TF-IDF")
    parser.add_argument("source", type=Path, help="Path to a directory with .txt files or a single .txt file")
    parser.add_argument("--top", type=int, default=200, help="Number of top terms to output")
    parser.add_argument("--min-df", type=int, default=3, help="Minimum number of documents a term must appear in")
    parser.add_argument("--max-df", type=float, default=0.5, help="Maximum document frequency fraction for a term")
    parser.add_argument("--max-ngram", type=int, default=2, help="Maximum n-gram length to consider")
    parser.add_argument(
        "--scoring",
        choices=["mean", "max", "sum"],
        default="mean",
        help="How to aggregate TF-IDF scores across documents",
    )
    parser.add_argument(
        "--extra-stop-words",
        nargs="*",
        default=[],
        help="Additional stop words to ignore (space separated)",
    )
    parser.add_argument(
        "--stop-words-file",
        type=Path,
        help="Path to a file with custom stop words, one per line",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=3,
        help="Minimum term length to keep after filtering",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to save the resulting terms and scores as CSV",
    )
    return parser.parse_args()


def load_custom_stopwords(path: Path | None) -> list[str]:
    if not path:
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    base_stopwords = merge_stopwords(args.extra_stop_words + load_custom_stopwords(args.stop_words_file))
    replacements = compile_replacements(build_default_replacements())

    documents = load_documents(args.source)
    processed_docs = [preprocess(doc, replacements, base_stopwords) for doc in documents]

    vectorizer = build_vectorizer(args)
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    terms = vectorizer.get_feature_names_out()
    scores = compute_scores(tfidf_matrix, args.scoring)

    filtered = filter_terms(terms, scores, args.min_length)
    filtered.sort(key=lambda pair: pair[1], reverse=True)
    top_terms = filtered[: args.top]

    for term, score in top_terms:
        print(f"{term}\t{score:.6f}")

    if args.output_csv:
        save_to_csv(top_terms, args.output_csv)
        print(f"Saved {len(top_terms)} terms to {args.output_csv}")


if __name__ == "__main__":
    main()
