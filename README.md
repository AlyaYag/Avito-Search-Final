# Поиск статей справки Авито по пользовательскому вопросу

Dense retrieval для RAG-пайплайна: по тексту вопроса пользователя найти 10 наиболее релевантных статей справки и вернуть их ранжированный список. Метрика — **MAP@10**.

## Задача

- **Корпус:** 793 статьи (`candidate_data/articles.f`) — заголовок + тело в HTML
- **Калибровка:** 500 размеченных запросов (`candidate_data/calibration.f`) с `ground_truth` — набором релевантных `article_id`
- **Тест:** 500 запросов без разметки (`candidate_data/test.f`) — для них нужно сгенерировать `answer.csv`
- **Ограничения:** без внешних LLM/API; только локальные open-source модели

## Что сделано

### Обработка HTML статей

Тела статей хранятся в сыром HTML с элементами Авито: изображения, спойлеры, табсеты (вкладки), вложенные таблицы, фактоиды, метки `<label>`, пустые теги. Применена двухстадийная очистка:

**Стадия 1 — BeautifulSoup** (`src/preprocess_html.py`, 12 шагов в конвейере):

1. `clean_images` — замена `<img>` на `alt`-текст (или удаление, если alt пуст)
2. `clean_breaks` — замена `<br>` на пробел
3. `clean_inputs` — удаление `<input>` (служебные элементы форм)
4. `clean_empty_labels` — удаление пустых `<label>`
5. `clean_links` — замена `<a>` на текст ссылки
6. `clean_headline_chunk` — разворачивание `<headline>`/`<chunk>` (кастомные теги Авито)
7. `clean_empty_tags` — удаление пустых `<p>`, `<div>`, `<span>`, `<strong>`, `<em>`
8. `clean_factoids` — разворачивание блоков `factoid` (вынос подсказок в текст)
9. `clean_spoilers` — замена `<div class="spoiler">` на заголовок спойлера (жирным) + содержимое
10. `clean_tabsets` — замена `<div class="tabset">` на заголовки вкладок (жирным) + содержимое панелей
11. `clean_nested_tables` — схлопывание вложенных таблиц: текст внутренней таблицы вставляется вместо неё
12. `clean_empty_tables` — удаление пустых таблиц

Результат: очищенный HTML без служебных элементов.

**Стадия 2 — html2text** (`src/html2text_stage.py`):

Конфигурация `html2text.HTML2Text`:
- `body_width=0` — без переноса строк
- `ignore_emphasis=True`, `ignore_links=True`, `ignore_images=True` — игнорирование форматирования
- `single_line_break=True` — компактный вывод
- `pad_tables=True` — читаемые таблицы
- `unicode_snob=True` — без entities

После конвертации дополнительное схлопывание пробелов: множественные пробелы → один, 3+ переводов строк → 2.

**Итог:** для каждой статьи доступны title (исходный заголовок) и body (очищенный текст).

### Признаки, модели и алгоритмы

**Типы признаков:**

- **Только заголовки статей (title)** — короткий ёмкий текст (средняя длина 29 символов). Выбран как основной источник, так как плотно передаёт тему статьи.
- **Тела статей (body)** — очищенный HTML-текст. Протестирован отдельно для сравнения.

**Модель эмбеддингов:**

`BAAI/bge-m3` — мультиязычный sentence transformer, 1024-мерные эмбеддинги, нормализация L2.


**Алгоритм ранжирования:**

Косинусная близость:

```python
scores = article_emb @ query_emb.T
top_k = article_ids[np.argsort(scores)[::-1][:10]]
```

Никаких дополнительных весов, реранжинга или фильтрации.

**Query rewriting (экспериментально):**

Модель: `Qwen/Qwen2.5-0.5B-Instruct` (500M параметров, CPU, `do_sample=False`, `temperature=None`, seed=42)

Few-shot промпт с 5 примерами нормализации шумных запросов поддержки:

> *«ой, извините, пожалуйста, я тут случайно удалила всё, что делать? помогите» → «восстановление данных после случайного удаления»*

Задача модели: выделить ключевую интенцию, убрав извинения, вежливость, лишние детали.

### Проверка качества на calibration.f

Процедура валидации (`src/validate.py`):

1. Загружаются эмбеддинги статей (из `data/*.parquet`)
2. Загружаются эмбеддинги калибровочных запросов (из `data/calibration_query_embeddings.parquet`)
3. Для каждого из 500 запросов:
   - Вычисляется косинусная близость со всеми 793 статьями
   - Выбирается top-10 по убыванию
   - Считается AP@10: precision на каждой релевантной позиции, усреднение по `min(|gt|, 10)`
4. MAP@10 = среднее AP@10 по всем запросам
5. Дополнительно логируются:
   - 10 худших запросов (AP=0) с текстом, ground truth и тем, что вернулось
   - 10 лучших запросов (AP=1.0) для проверки адекватности
   - 10 worst articles по числу false positive retrievals
   - 10 worst articles по числу пропусков (relevant, но не найдены)
   - AP distribution по корзинам 0.0, 0.1, ..., 1.0

Лог сохраняется в `results/runs/{method}.txt`, per-query AP — в `results/runs/{method}_per_query_ap.npy`.

Всего проверено 4 конфигурации:

| Метод | MAP@10 | AP=0 |
|-------|--------|------|
| Dense retrieval на заголовках | **0.1431** | 322 (64.4%) |
| Dense retrieval на телах статей | 0.0958 | 380 (76.0%) |
| Query rewriting → заголовки | 0.0478 | 430 (86.0%) |
| Query rewriting → тела статей | 0.0784 | 415 (83.0%) |

### Анализ ошибок и что с ними сделано

**Тип 1: 64% запросов имеют AP=0**

Модель не находит ни одной релевантной статьи в top-10. Типичный пример:

```
query: Как передать товар через службу авито
expected: 1909 «Отправить заказ», 4234 «Как продавать и покупать с доставкой»
got:      1910 «Проверить, что товар заказали через Авито», 3641 «...», 3889 «...», ...
```

Причина: bge-m3 находит статьи, содержащие те же слова («Авито», «доставка», «товар»), но не понимает связь между «передать товар» и «отправить заказ». Статья «Отправить заказ» (article_id=1909) не попадает в top-10, хотя это точный ответ.

*Что сделано:* попытка query rewriting для схлопывания синонимов не улучшила результат (MAP упал с 0.1431 до 0.0478) — модель 0.5B слишком агрессивно обобщает и теряет специфику запроса.

**Тип 2: Статья 4219 «Покупателю» никогда не находится**

Релевантна для 129 запросов, но не найдена ни разу (recall = 0.00). Заголовок слишком общий — «Покупателю» — без указания темы. Пользователи не спрашивают «вопрос покупателю», они спрашивают «как вернуть деньги», «продавец не отправляет» и т.д.

*Что сделано:* на данном этапе ничего. Гипотеза: если добавить body статьи в эмбеддинг (с keyphrase-извлечением или weighted pooling), recall для таких статей может вырасти.

**Тип 3: Статья 2943 «Не могу вывести деньги за доставку» — источник ложных срабатываний**

Статья извлечена 195 раз, но релевантна только для 1 запроса (194 false positive). Причина: заголовок содержит слова «деньги» и «доставка», которые встречаются во множестве запросов про возвраты и оплату.

*Что сделано:* данный тип ошибок не адресован в текущем решении. Возможные подходы: IDF-взвешивание (BM25 вместо dense) или реранжинг кросс-энкодером.

**Тип 4: Query rewriting ухудшил качество**

MAP@10 упал с 0.1431 до 0.0478 на заголовках и до 0.0784 на телах. Визуальный анализ сгенерированных запросов показал, что модель 0.5B часто:
- Удаляет ключевые слова (например, «Авито Доставка» → «доставка»)
- Генерирует неточные обобщения (например, «как активировать промокод на доставку за 500р от теле2» → «промокод на доставку» — теряется оператор «активировать»)

*Что сделано:* решено не использовать рерайтинг в финальном `answer.csv`. Для `answer.csv` применяется dense retrieval на заголовках — best-performing метод.

**Итоговый вывод:** dense retrieval на заголовках (bge-m3) даёт MAP@10 = 0.143. Основная проблема — 64% запросов не получают релевантных статей в топ-10. Для улучшения требуется комбинация с sparse retrieval (BM25) или Doc2Query-расширение индекса.

## Воспроизведение

### Требования

- Docker (проверить: `docker ps`)
- ~3-4 ГБ свободного места под модель bge-m3 (+ ~1.5 ГБ для Qwen, если нужен рерайтинг)

### Воспроизведение лучшего результата (MAP@10 = 0.1431)

Лучший результат даёт **dense retrieval на заголовках** (`dense_title`). Минимальный набор шагов:

```bash
# 1. Сборка образа
docker build -t avito-search .

# 2. Полный пайплайн (предобработка + эмбеддинги статей + эмбеддинги запросов)
docker run --rm \
  -v "$PWD/candidate_data:/app/candidate_data" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avito-search \
  python run_pipeline.py

# 3. Валидация на calibration.f (среди прочего считает dense_title)
docker run --rm \
  -v "$PWD/candidate_data:/app/candidate_data" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avito-search \
  python run_search.py

# 4. Генерация answer.csv для test.f (использует title embeddings — best method)
docker run --rm \
  -v "$PWD/candidate_data:/app/candidate_data" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avito-search \
  python src/generate_answer.py
```

После выполнения:
- `results/runs/dense_title.txt` — лог валидации с MAP@10 = 0.1431
- `results/answer.csv` — финальный ответ для test.f

**Важно:** `run_pipeline.py` выполняет в том числе энкодинг тел статей и рерайтинг запросов (нужны для экспериментальных методов, но не для лучшего результата). Если нужно только `dense_title`, закомментируйте строки `encode_article_bodies` и `rewrite_queries` в `run_pipeline.py` — это ускорит запуск.

**Валидация только `dense_title`** (без запуска всего `run_search.py`):

```bash
docker run --rm \
  -v "$PWD/candidate_data:/app/candidate_data" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avito-search \
  python src/search.py --method dense_title --articles-path data/title_embeddings.parquet
```

### Dockerfile

```bash
docker build -t avito-search .
```

### Два режима запуска

Система разделена на два независимых entry point, каждый из которых запускается через `docker run`. Это позволяет не пересчитывать эмбеддинги статей при каждом изменении подхода к поиску.

---

#### Режим 1: `run_pipeline.py` — Предобработка + эмбеддинги статей

Запускается **один раз** (или при изменении обработки статей/модели):

1. Очистка HTML → plain text
2. Энкодинг заголовков статей → `data/title_embeddings.parquet`
3. Энкодинг тел статей → `data/article_body_embeddings.parquet`
4. Энкодинг калибровочных запросов → `data/calibration_query_embeddings.parquet`
5. Рерайтинг запросов Qwen + энкодинг → `data/rewritten_queries.parquet`, `data/calibration_query_embeddings_rewritten.parquet`

```bash
docker run --rm \
  -v "$PWD/candidate_data:/app/candidate_data" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avito-search \
  python run_pipeline.py
```

**Если рерайтинг не нужен**, можно раскомментировать соответствующую строку в `run_pipeline.py`.

---

#### Режим 2: `run_search.py` — Валидация всех методов на calibration.f

Запускается **многократно** для экспериментов с подходами к ранжированию (использует уже готовые эмбеддинги из `data/`):

1. Энкодинг калибровочных запросов
2. Валидация всех 4 методов с расчётом MAP@10
3. Логи в `results/runs/{method}.txt` — худшие/лучшие запросы, AP distribution, worst articles

```bash
docker run --rm \
  -v "$PWD/candidate_data:/app/candidate_data" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avito-search \
  python run_search.py
```

#### Генерация финального answer.csv

```bash
docker run --rm \
  -v "$PWD/candidate_data:/app/candidate_data" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avito-search \
  python src/generate_answer.py
```

Результат: `results/answer.csv` (500 строк, колонки `query_id`, `answer`).

**Важно:** `answer.csv` всегда генерируется через `dense_title` (best method), независимо от того, какие методы валидировались и какие эмбеддинги есть в `data/`.

### Без Docker

```bash
pip install torch sentence-transformers pandas pyarrow numpy scikit-learn beautifulsoup4 html2text
python run_pipeline.py
python run_search.py
python src/generate_answer.py
```

Для CPU-версии torch используйте `--extra-index-url https://download.pytorch.org/whl/cpu`.
