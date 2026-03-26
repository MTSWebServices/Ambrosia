# Ambrosia

Python-библиотека для A/B-тестирования: дизайн экспериментов, разбиение на группы, оценка эффекта. Поддержка pandas и PySpark.

## Команды

```bash
# Установка
make install                  # poetry install + extras

# Тесты
make test                     # pytest
poetry run pytest tests/ -x   # с остановкой на первом падении
poetry run pytest tests/test_designer.py -x  # конкретный файл

# Линтеры (проверка)
make lint                     # isort + black + pylint + flake8

# Форматирование (авто-исправление)
make autoformat               # isort + black

# Coverage
make coverage
```

## Архитектура

Три основных модуля образуют пайплайн:
- `ambrosia/designer/` — расчёт параметров эксперимента (размер выборки, MDE, мощность)
- `ambrosia/splitter/` — разбиение пользователей на группы (simple, hash, metric, stratification)
- `ambrosia/tester/` — оценка эффекта и статзначимости (t-test, Mann-Whitney, Wilcoxon, bootstrap)

Предобработка:
- `ambrosia/preprocessing/` — агрегация, outlier removal, Box-Cox, Log, CUPED, ML variance reduction

Ядро:
- `ambrosia/tools/` — абстрактные классы, стат. критерии, KNN, утилиты
- `ambrosia/spark_tools/` — PySpark-реализации (опциональная зависимость)

### Иерархия абстракций

```
ABToolAbstract          — базовый класс для Designer, Splitter, Tester
AbstractFittableTransformer — базовый для трансформеров (BoxCox, Log, Robust, IQR, Aggregate, Cuped)
AbstractVarianceReducer     — базовый для Cuped, MultiCuped, MLVarianceReducer
ABStatCriterion             — базовый для TtestIndCriterion, MannWhitneyCriterion и др.
```

Каждый основной класс (Designer, Splitter, Tester) реализует паттерн:
- Конфигурация через `set_*()` методы или конструктор
- Запуск через `run()` метод
- Поддержка YAML-сериализации

## Код-стайл

- **Line length:** 120 символов (black, isort, flake8 — всё настроено на 120)
- **Formatter:** black
- **Import sort:** isort (trailing comma, parentheses, case-sensitive)
- **Docstrings:** NumPy convention
- **Лицензионный заголовок** в каждом .py файле:
  ```python
  #  Copyright 2022 MTS (Mobile Telesystems)
  #
  #  Licensed under the Apache License, Version 2.0 (the "License");
  #  ...
  ```
- **Type hints:** используются через `ambrosia/types.py` — единый модуль типов
- **Flake8 игнорирует:** D200, D205, D400, D105, D100, E203, W503
- **Pylint:** конфигурация в `.pylintrc`, игнорирует `tests/`

## Тестирование

- Фреймворк: pytest
- Маркеры: `@pytest.mark.unit`, `@pytest.mark.smoke`
- Фикстуры: `tests/conftest.py` (включая local Spark session)
- Тестовые данные: `tests/test_data/`
- Паттерн именования: `test_*.py`, функции `test_*`

## Важные соглашения

- PySpark — опциональная зависимость (`pip install ambrosia[spark]`). Импорт Spark-модулей защищён через `ambrosia/tools/import_tools.py`
- KNN использует nmslib (primary) с fallback на hnswlib (для macOS ARM)
- Python 3.9–3.13, PySpark >= 3.4
- Управление зависимостями: Poetry (pyproject.toml)
- CI: GitHub Actions (lint + test matrix по версиям Python)
