# Поиск статей справки по пользовательскому вопросу

## Задача

Для каждого запроса из test.f вернуть ранжированный список article_id (top-10) —
самые подходящие статьи справки Авито. Метрика — MAP@10.

## Данные

- `candidate_data/articles.f` — 793 статьи (title + body в HTML)
- `candidate_data/calibration.f` — 500 размеченных запросов (query_id, query_text, ground_truth)
- `candidate_data/test.f` — 500 запросов без разметки

## Решение (текущий baseline)

- Модель: `BAAI/bge-m3` (dense retrieval)
- Используются только заголовки статей
- Query encoding с инструкцией: `"Represent this sentence for searching relevant passages: "`
- Оценка на calibration.f с логированием в `results/runs/`

## Запуск

### Требования

- Docker (проверить: `docker ps`)

### Полный пайплайн (предобработка → эмбеддинги статей)

```bash
docker build -t avito-search .
docker run --rm \
  -v "$PWD/candidate_data:/app/candidate_data" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avito-search \
  python run_pipeline.py
```

### Поиск (энкод запросов + ранжирование + answer.csv)

```bash
docker run --rm \
  -v "$PWD/candidate_data:/app/candidate_data" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avito-search \
  python run_search.py
```

Результаты:
- `results/answer.csv` — ответы для test.f
- `results/calibration_query_embeddings.parquet` — эмбеддинги запросов
- `results/runs/{method}.txt` — лог валидации (MAP@10, худшие запросы/статьи)

### Пошаговый запуск (для отладки)

```bash
# 1. Очистка HTML
docker run --rm -v "$PWD/candidate_data:/app/candidate_data" avito-search \
  python src/preprocess_html.py candidate_data/articles.f candidate_data/articles_cleaned.f

# 2. HTML → текст
docker run --rm -v "$PWD/candidate_data:/app/candidate_data" avito-search \
  python src/html2text_stage.py candidate_data/articles_cleaned.f candidate_data/articles_text.f

# 3. Эмбеддинги статей
docker run --rm \
  -v "$PWD/candidate_data:/app/candidate_data" \
  -v "$PWD/data:/app/data" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avito-search \
  python src/encode_articles.py

# 4. Эмбеддинги запросов
docker run --rm \
  -v "$PWD/candidate_data:/app/candidate_data" \
  -v "$PWD/results:/app/results" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avito-search \
  python src/encode_queries.py

# 5. Ранжирование + валидация + answer.csv
docker run --rm \
  -v "$PWD/candidate_data:/app/candidate_data" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avito-search \
  python src/search.py
```

## Структура проекта

```
├── Dockerfile
├── run_pipeline.py          # entry point: preprocess → encode articles
├── run_search.py            # entry point: encode queries → search → answer.csv
├── src/
│   ├── encode_articles.py   # эмбеддинги заголовков статей
│   ├── encode_queries.py    # эмбеддинги запросов
│   ├── search.py            # ранжирование + валидация + answer.csv
│   ├── validate.py          # MAP@10 + анализ ошибок
│   ├── rankers.py           # функции ранжирования
│   ├── utils.py             # AP@10, MAP@10
│   ├── preprocess_html.py   # очистка HTML (beautifulsoup4)
│   └── html2text_stage.py   # HTML → текст (html2text)
├── candidate_data/          # исходные данные (монтируются)
├── data/                    # эмбеддинги статей (создаются)
└── results/                 # эмбеддинги запросов, answer.csv, логи
```

## Добавление нового метода ранжирования

1. Написать функцию в `src/rankers.py`:
   ```python
   def my_ranker(article_emb, article_ids, query_emb, top_k=10):
       # ... ваш алгоритм
       return rankings  # list[list[int]]
   ```
2. Запустить валидацию через `src/validate.py`:
   ```python
   from src.validate import validate
   from src.rankers import my_ranker

   validate(
       article_emb_path="data/article_embeddings.parquet",
       query_emb_path="results/calibration_query_embeddings.parquet",
       ranker=my_ranker,
       method_name="my_method",
   )
   ```
3. Лог сохранится в `results/runs/my_method.txt`
