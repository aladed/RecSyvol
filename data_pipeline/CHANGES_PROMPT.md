# Промпт для Gemini 3 Pro: Изменения в пайплайне обработки T-ECD датасета

## Контекст задачи

Я работаю над пайплайном обработки данных для рекомендательной системы на датасете T-ECD (e-commerce dataset). Изначально пайплайн был настроен на полную версию датасета (`dataset/full`), но теперь нужно переделать его под малую версию (`dataset/small`), которая имеет другую структуру расположения файлов и другой набор данных.

## Изменения в структуре данных

### Старая структура (dataset/full):
```
t_ecd_small/
  dataset/
    full/
      brands.pq
      users.pq
      marketplace/
        items.pq
        events/
          01307.pq
          01308.pq
      retail/
        items.pq
        events/
          01307.pq
          01308.pq
      offers/
        items.pq
        events/
          01307.pq
          01308.pq
      reviews/
        01307.pq
        01308.pq
```

### Новая структура (dataset/small):
```
t_ecd_small/
  dataset/
    brands.pq
    users.pq
    marketplace/
      items.pq
      events/
        [227 файлов: 01082.pq - 01308.pq]
    retail/
      items.pq
      events/
        [227 файлов: 01082.pq - 01308.pq]
    offers/
      items.pq
      events/
        [227 файлов: 01082.pq - 01308.pq]
    reviews/
      [227 файлов: 01082.pq - 01308.pq]
```

### Ключевые отличия:
1. **Убрана подпапка `full/`**: данные лежат прямо в `dataset/`, а не в `dataset/full/`
2. **Больше дней событий**: в small версии 227 дней (01082-01308), а не только 2 дня (01307-01308)
3. **Меньше данных**: small версия содержит меньше пользователей и событий, но структура файлов идентична

## Изменения в коде пайплайна

### 1. Базовые пути (Ячейка 2)

**Было:**
```python
base_path = "t_ecd_small/dataset/full"
```

**Стало:**
```python
base_path = "t_ecd_small/dataset"
```

**Причина**: Данные теперь лежат в `dataset/`, без подпапки `full/`

---

### 2. Функция загрузки данных (Ячейка 3) - КРИТИЧНОЕ ИЗМЕНЕНИЕ

**Было:**
```python
def safe_data_loader():
    base_path = "t_ecd_small/dataset/full"
    
    # События загружались только из одного дня
    for domain, (path, sample_rate) in events_config.items():
        events = pl.read_parquet(f"{base_path}/{path}/01307.pq").sample(sample_rate)
        events_data[domain] = events
```

**Проблема**: В новом датасете файл `01307.pq` может быть пустым или содержать очень мало данных, что приводило к:
- `marketplace: (0, 6)` - пустой датафрейм
- `retail: (0, 6)` - пустой датафрейм
- `offers: (0, 4)` - пустой датафрейм
- `reviews: (0, 5)` - пустой датафрейм

**Стало:**
```python
def safe_data_loader():
    from pathlib import Path
    base_path = Path("t_ecd_small/dataset")
    
    # 1. Статические файлы (без изменений в логике)
    static_files = {
        'users': 'users.pq',
        'brands': 'brands.pq', 
        'marketplace_items': 'marketplace/items.pq',
        'retail_items': 'retail/items.pq',
        'offers_items': 'offers/items.pq'
    }
    
    for name, rel_path in static_files.items():
        static_data[name] = pl.read_parquet(base_path / rel_path)
    
    # 2. События: загружаем ВСЕ дни и сэмплируем строки
    events_config = {
        'marketplace': ('marketplace/events', 0.05),  # 5% от всех событий
        'retail': ('retail/events', 0.05), 
        'offers': ('offers/events', 0.1),            # событий меньше, можно больше
        'reviews': ('reviews', 0.1)                  # Без папки events!
    }
    
    for domain, (rel_dir, sample_rate) in events_config.items():
        domain_dir = base_path / rel_dir
        files = sorted(domain_dir.glob("*.pq"))  # Находим ВСЕ файлы дней
        
        if not files:
            print(f"❌ {domain}: нет файлов в {domain_dir}")
            continue

        # Ленивая загрузка всех файлов и сэмплирование по строкам
        lazy_df = pl.scan_parquet([str(p) for p in files])
        events = lazy_df.sample(fraction=sample_rate).collect()
        events_data[domain] = events
```

**Ключевые изменения:**
- Используется `Path` для работы с путями
- **Глобальный поиск всех файлов**: `domain_dir.glob("*.pq")` находит все 227 дней
- **Ленивая загрузка всех файлов**: `pl.scan_parquet([str(p) for p in files])` читает все дни сразу
- **Сэмплирование по строкам**: `lazy_df.sample(fraction=sample_rate)` берёт случайную выборку из всех событий всех дней
- Это даёт репрезентативную выборку из всего датасета, а не из одного дня

---

### 3. Конфигурация путей (Ячейка 4)

**Было:**
```python
RAW_DATA_PATH = Path("t_ecd_small/dataset/full")
```

**Стало:**
```python
RAW_DATA_PATH = Path("t_ecd_small/dataset")
```

**Причина**: Убрана подпапка `full/`

---

### 4. Анализ файлов (Ячейка 5)

**Было:**
```python
base_path = "t_ecd_small/dataset/full"
```

**Стало:**
```python
base_path = "t_ecd_small/dataset"
```

**Примечание**: Функция `analyze_all_files()` анализирует конкретные файлы (01307.pq, 01308.pq), которые существуют в новом датасете, поэтому работает корректно. Однако она анализирует только 2 дня из 227 доступных.

---

### 5. Построение словаря (Ячейка 7)

**Было:**
```python
RAW_DIR = Path("t_ecd_small/dataset/full")
```

**Стало:**
```python
RAW_DIR = Path("t_ecd_small/dataset")
```

**Логика функции `build_vocab()` не изменилась:**
- Читает `marketplace/items.pq`, `retail/items.pq`, `offers/items.pq`, `brands.pq`
- Создаёт токены с префиксами: `MP_`, `RT_`, `OF_`, `BR_`
- Сохраняет в `t_ecd_small/dataset/processed/vocab.parquet`

**Причина**: Структура файлов items и brands идентична в обеих версиях датасета

---

### 6. Обработка шардов (Ячейка 8) - КРИТИЧНОЕ ИЗМЕНЕНИЕ

**Было:**
```python
RAW_DIR = Path("t_ecd_small/dataset/full")
```

**Стало:**
```python
RAW_DIR = Path("t_ecd_small/dataset")
```

**Функция `get_domain_plan()` уже была правильно написана:**
```python
def get_domain_plan(domain_folder: Path, domain_prefix: str, vocab_lazy):
    # Для reviews файлы лежат прямо в папке, без подпапки events
    if "reviews" in str(domain_folder):
        file_paths = list(domain_folder.glob("*.pq"))  # reviews/*.pq
    else:
        file_paths = list((domain_folder / "events").glob("*.pq"))  # domain/events/*.pq
    
    # Сканируем все файлы и объединяем
    lazy_frames = [pl.scan_parquet(str(p)) for p in file_paths]
    q = pl.concat(lazy_frames)  # Объединяет ВСЕ дни
    
    # Маппинг через join с vocab
    q = q.select([
        pl.col("user_id"),
        pl.col("timestamp"),
        (pl.lit(domain_prefix) + pl.col(entity_col).cast(pl.Utf8)).alias("token_str")
    ])
    
    q = q.join(
        vocab_lazy.select(["token_str", "token_id"]),
        on="token_str",
        how="left"
    ).with_columns(
        pl.col("token_id").fill_null(4).cast(pl.UInt32)  # UNK token = 4
    )
```

**Ключевые особенности:**
- **Автоматически находит все дни**: `glob("*.pq")` находит все 227 файлов
- **Обрабатывает все дни**: `pl.concat(lazy_frames)` объединяет все события
- **Использует join вместо map_dict**: для больших словарей (30M+ токенов) join эффективнее
- **Правильная обработка reviews**: файлы лежат в `reviews/`, а не в `reviews/events/`

**Функция `process_shards()` обрабатывает:**
- Все 227 дней событий для каждого домена
- Разбивает пользователей на 50 шардов по хешу `user_id`
- Создаёт последовательности взаимодействий для каждого пользователя
- Сохраняет в `t_ecd_small/dataset/processed/shards/shard_*.parquet`

---

## Технические детали изменений

### Использование Path вместо строк
**Было:**
```python
f"{base_path}/{path}/01307.pq"
```

**Стало:**
```python
base_path = Path("t_ecd_small/dataset")
base_path / rel_path
```

**Преимущества:**
- Кроссплатформенность (Windows/Unix)
- Безопасность путей
- Удобство работы с glob

### Ленивая загрузка всех дней
**Было:**
```python
pl.read_parquet(f"{base_path}/{path}/01307.pq")  # Один файл
```

**Стало:**
```python
files = sorted(domain_dir.glob("*.pq"))  # Все файлы
lazy_df = pl.scan_parquet([str(p) for p in files])  # Ленивая загрузка
events = lazy_df.sample(fraction=sample_rate).collect()  # Сэмплирование
```

**Преимущества:**
- Обрабатывает все дни автоматически
- Не загружает все данные в память сразу (lazy evaluation)
- Репрезентативная выборка из всего датасета

### Join вместо map_dict для больших словарей
**Было (в старой версии):**
```python
vocab_map = dict(zip(vocab_df["token_str"], vocab_df["token_id"]))  # 30M+ записей в памяти
q = q.with_columns(
    pl.col("token_key").map_dict(vocab_map, default=4)
)
```

**Стало:**
```python
vocab_lazy = pl.scan_parquet(VOCAB_PATH)  # Ленивый словарь
q = q.join(
    vocab_lazy.select(["token_str", "token_id"]),
    on="token_str",
    how="left"
).with_columns(
    pl.col("token_id").fill_null(4).cast(pl.UInt32)
)
```

**Преимущества:**
- Не загружает весь словарь в память
- Polars оптимизирует join автоматически
- Работает эффективно для словарей 30M+ токенов

---

## Результаты изменений

### До изменений:
- Загружались события только из одного дня (01307.pq)
- В новом датасете этот день мог быть пустым → пустые датафреймы
- Не использовались все доступные данные (227 дней)

### После изменений:
- Загружаются события из всех 227 дней
- Репрезентативная выборка из всего датасета
- Корректная работа с новым small датасетом
- Эффективное использование памяти через lazy evaluation

---

## Структура пайплайна после изменений

1. **Ячейка 0-1**: Загрузка датасета с HuggingFace (без изменений)
2. **Ячейка 2**: Базовый путь → `t_ecd_small/dataset`
3. **Ячейка 3**: Загрузка данных → читает все дни, сэмплирует строки
4. **Ячейка 4**: Конфигурация путей → `t_ecd_small/dataset`
5. **Ячейка 5**: Анализ файлов → работает с новыми путями
6. **Ячейка 6**: Tokenizer класс (без изменений)
7. **Ячейка 7**: Построение словаря → работает с новыми путями
8. **Ячейка 8**: Обработка шардов → обрабатывает все 227 дней

---

## Вопросы для обсуждения с Gemini 3 Pro

1. **Оптимизация сэмплирования**: Сейчас используется `sample(fraction=0.05)` для marketplace/retail. Это даёт репрезентативную выборку, но может быть медленным на больших данных. Есть ли более эффективные способы?

2. **Обработка reviews**: Reviews используют `brand_id` вместо `item_id`. Правильно ли это для рекомендательной системы, или нужно учитывать и item_id из reviews?

3. **Разделение train/test**: В коде есть `TEST_DAYS_CUTOFF = 2`, но он не используется. Нужно ли разделять данные по времени для валидации?

4. **Масштабирование**: При обработке всех 227 дней может быть проблема с памятью. Стоит ли обрабатывать дни батчами?

5. **Валидация данных**: Нужно ли добавить проверки на:
   - Пустые файлы дней
   - Отсутствующие колонки
   - Некорректные типы данных

---

## Заключение

Все изменения направлены на адаптацию пайплайна под новую структуру small датасета:
- Убрана подпапка `full/` из всех путей
- Изменена логика загрузки событий: теперь читаются все дни, а не один
- Используется lazy evaluation для эффективной работы с большими данными
- Join вместо map_dict для работы с большими словарями

Пайплайн теперь корректно работает с новым датасетом и использует все доступные данные (227 дней вместо 2).

