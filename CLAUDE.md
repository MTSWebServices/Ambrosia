# Ambrosia

A/B testing framework for experiment design, group splitting, and results evaluation.
Supports both pandas and Spark DataFrames.

## Commands

```bash
make install      # create .venv via Poetry (poetry install --all-extras)
make test         # run pytest with coverage
make lint         # isort + black + pylint + flake8 (checks only)
make autoformat   # isort + black (fix in place)
make clean        # remove .venv, build artifacts, reports/
```

Single test: `PYTHONPATH=. pytest tests/path/test_file.py::test_fn`

Line length: **120**.

## Architecture

### Three-stage pipeline

`Designer` → `Splitter` → `Tester` are independent, stateless-ish classes.
No shared state between stages; each takes a DataFrame and parameters.

### Pandas/Spark dispatch

Never subclass for pandas vs. Spark. Instead use `DataframeHandler` or the
free function `choose_on_table(alternatives, dataframe)` in
`ambrosia/tools/ab_abstract_component.py`:

```python
choose_on_table([pandas_func, spark_func], dataframe)
```

`DataframeHandler._handle_cases` / `_handle_on_table` wrap this pattern for
method dispatch in handlers (e.g. `TheoryHandler`, `EmpiricHandler`).

### ABMetaClass

`ABMetaClass(ABCMeta, YAMLObjectMetaclass)` in `ab_abstract_component.py`
resolves the metaclass conflict between `ABCMeta` and PyYAML's
`YAMLObjectMetaclass`. Any class that inherits from `ABToolAbstract` **and**
needs YAML serialization must set `metaclass=ABMetaClass`.

### ABToolAbstract._prepare_arguments()

Constructor args are "saved" defaults; `run()` args can override them at
call time. `_prepare_arguments` resolves the priority:
run-time arg → constructor arg → `ValueError` if both are None.

```python
chosen = _prepare_arguments({"alpha": [self._alpha, given_alpha]})
```

### Stat criteria strategy pattern

Hierarchy: `StatCriterion` (abstract, just `calculate_pvalue`) →
`ABStatCriterion` (adds `calculate_effect`, `calculate_conf_interval`,
`get_results`).

Concrete implementations in `ambrosia/tools/stat_criteria.py`:
`TtestIndCriterion`, `TtestRelCriterion`, `MannWhitneyCriterion`,
`WilcoxonCriterion`.

`Tester` dispatches by string alias via `AVAILABLE_AB_CRITERIA` dict — duck
typing, not isinstance checks. To add a criterion: subclass `ABStatCriterion`,
set `alias` and `implemented_effect_types` class attributes, register in the
dict.

### Preprocessor chain

`Preprocessor` (pandas only) uses method chaining — each method returns
`self`. Each step appends a fitted `AbstractFittableTransformer` to
`self.transformers`. The transformer list supports serialization
(`store_transformations` / `load_transformations` → JSON) and replay
(`apply_transformations`) for consistent train/test preprocessing.

### Theoretical vs empirical design

Two design philosophies plug into the same `SimpleDesigner` interface:

- **Theoretical** (`TheoryHandler`): closed-form power/sample-size formulas
- **Empirical** (`EmpiricHandler`): bootstrap/simulation-based estimates

Both implement `size_design`, `effect_design`, `power_design` and dispatch
pandas vs. Spark internally via `DataframeHandler`.
