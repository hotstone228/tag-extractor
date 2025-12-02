# TF-IDF Tag Extractor

Скрипт `extract_tags.py` собирает кандидатов в теги из корпуса текстов с помощью TF‑IDF.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

1. Сложите статьи в папку с `.txt` файлами.
2. Выполните команду:

```bash
python extract_tags.py --input-dir path/to/texts \
  --output candidates.csv \
  --top-n 300 \
  --min-df 3 \
  --max-df 0.5 \
  --ngram-max 3 \
  --lemmatize \
  --stopword "название_компании"
```

Ключевые параметры:

- `--lemmatize` — включает лемматизацию русских слов через `pymorphy2` (опционально).
- `--ngram-max` — собирает биграммы/триграммы для более информативных фраз.
- `--stopword` — можно передать несколько раз для добавления доменных стоп-слов.

Результат записывается в CSV с колонками `term,score`. Термины предварительно очищаются от очевидного шума (цифровые строки, IP/URL, хэши).
