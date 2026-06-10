Release Notes
=============

Version 0.5.3 (unreleased)
---------------------------

**New Features:**

* Sample Ratio Mismatch (SRM) check. A broken assignment, filtering or logging
  pipeline often shows up as group sizes that deviate from the planned split -
  and then any test result on such groups cannot be trusted. The ``Tester`` now
  verifies the observed group sizes with a chi-square test (strict ``0.0005``
  level) before evaluating the results and emits a warning when a mismatch is
  detected. For intentionally unequal splits pass ``srm_expected_ratios``
  (e.g. ``{"A": 0.9, "B": 0.1}``); disable with ``check_srm=False``. The same
  diagnostic is available standalone as ``ambrosia.tools.srm.check_srm`` for
  pandas and Spark dataframes.

**Bug Fixes:**

* ``Tester`` and the standalone ``test`` function no longer fail when
  ``first_type_errors`` is not provided explicitly: the documented default
  ``0.05`` is now applied.

* Fixed binary experiment design with Bayesian intervals: the conjugate prior
  parameters (``n_success_conjugate``, ``n_failure_conjugate``) for
  ``interval_type="bayes_beta"`` now reach the interval computation in size,
  effect and power design instead of raising ``TypeError``. Thanks to
  Artem Vasin for the original fix (#45).

* Fixed one-sided alternatives in the binary design power table: confidence
  bounds are now built with the correct shape for simulation matrices, so
  ``alternative="greater"``/``"less"`` no longer fails with a shape error.

* Keyword arguments that collide with internally derived confidence interval
  arguments (e.g. ``confidence_level``) now raise a clear ``ValueError``
  instead of silently overriding the design parameters.

Version 0.5.2 (01.06.2026)
---------------------------

**New Features:**

* The ``Tester`` now offers several ways to correct p-values when an experiment
  evaluates many metrics or many groups at once, not just Bonferroni. Running a
  lot of comparisons together inflates the chance of a false positive, and a
  correction keeps that chance under control. The ``correction_method`` argument
  now accepts:

  * ``"bonferroni"`` (the default, unchanged), ``"holm"``, ``"holm-sidak"``,
    ``"sidak"``, ``"hommel"`` and ``"simes-hochberg"`` - methods that limit the
    probability of making *any* false positive. ``"holm"`` and the others are
    less conservative than Bonferroni, so you keep more statistical power.
  * ``"fdr_bh"`` (Benjamini-Hochberg) and ``"fdr_by"`` (Benjamini-Yekutieli) -
    methods that control the false discovery rate, i.e. the expected share of
    false positives among the metrics you call significant. A popular choice
    when a dashboard tracks many metrics.

  Existing code keeps working unchanged: Bonferroni stays the default and
  ``correction_method=None`` turns correction off. Friendly aliases such as
  ``"benjamini-hochberg"`` are also accepted. For ``"bonferroni"`` and
  ``"sidak"`` the confidence intervals are widened to match; the other methods
  adjust the p-values only.

**Internal:**

* Added ``ambrosia/tools/multitest.py`` with native p-value adjustment
  procedures, cross-validated against ``statsmodels`` and R's ``p.adjust``.

Version 0.5.1 (26.03.2026)
---------------------------

**New Features:**

* Custom metric functions in ``Tester``: new ``metric_funcs`` parameter allows
  passing arbitrary callables instead of column names. Works with ``theory``
  and ``empiric`` methods. Functions passed to ``run()`` override those set
  in the constructor.

* ``LinearizationTransformer`` for ratio metrics (e.g. revenue/orders).
  Linearizes metric via ``linearized_i = numerator_i - ratio * denominator_i``,
  where ``ratio`` is estimated on reference data during ``fit()``.

* ``Preprocessor.linearize()`` integrates linearization into the existing
  chain architecture with full serialization and replay support.

**Bug Fixes:**

* Pinned ``setuptools>=65.0.0, <82.0.0`` to fix ``pkg_resources`` removal
  in setuptools 82 that broke ``pip install ambrosia`` due to hyperopt dependency.

**Internal:**

* Updated publish workflow to use ``PYPI_TOKEN_V2``

* Added CLAUDE.md to ``.gitignore``


Version 0.5.0 (06.01.2025)
---------------------------

**Breaking Changes:**

* Minimum Python version raised to 3.9 (dropped support for 3.7, 3.8)

* Minimum PySpark version raised to 3.4 (dropped support for 3.2, 3.3)

**New Features:**

* Added support for Python 3.11, 3.12, 3.13

**Bug Fixes:**

* Added hnswlib as fallback for nmslib on macOS ARM (fixes segfault in metric split)

**Dependencies:**

* Updated numpy to >=1.24.0, <3.0.0

* Updated pandas to >=1.5.0, <3.0.0

* Updated scipy to >=1.10.0

* Updated scikit-learn to >=1.3.0

* Updated nmslib to >=2.1.0

* Added hnswlib >=0.7.0 as alternative KNN backend

* Updated catboost to >=1.2.0

* Updated other dependencies for Python 3.12/3.13 compatibility

**Internal:**

* Replaced deprecated ``pkg_resources`` with ``importlib.metadata``

* Updated CI/CD to test Python 3.9-3.13

* Updated GitHub Actions to v4/v5


Version 0.4.1 (21.04.2023)
---------------------------

Hotfix for pyspark import in spark criteria.

Version 0.4.0 (21.04.2023)
---------------------------

* Documentation and usage examples have been substantially reworked and updated. 

* The ``Designer`` class and design methods functionality is updated. 
  
  * Empirical design now supports the choice of hypothesis alternative and group ratio parameter
  
  * Look of resulting tables with calculated parameters is unified for all design methods
  
  * Changed multiprocessing strategy for bootstrap criterion

* The ``Tester`` class functionality is updated. 

  * Spark data support for the ``Tester`` class is added. Independent t-test is available now

  *  Bootstrap criterion can now return deterministic output using a ``random_seed`` parameter

  * Paired bootstrap criterion is now available

  * MHC now is optional and takes into account the number of passed metrics

  *  ``first_errors`` parameter renamed to ``first_type_errors``

* ``pyspark`` package now is optional and could be installed using ``pip`` extras.

* Fixed a set of bugs.


Version 0.3.0 (15.02.2023)
---------------------------

* The ``Designer`` class and design methods functionality is updated. 

  * Theoretical design now supports the choice of hypothesis alternative and group ratio parameter 

  * These calculations now use Statsmodels solvers

  * Experimental parameters for binary data can now also be theoretically designed using both 
    the asin variance-stabilizing transformation and the normal approximation

* All preprocessor classes, except for the ``Preprocessor``, have changed their api and have updated functionality

  * Preprocessing classes now use ``fit`` and ``transform`` methods to get transformation parameters 
    and apply transformation on pandas tables

  * Fitted classes now can now be saved and loaded from json files

  * Table column names used when fitting class instances are now strictly fixed in instance attributes

* The ``Preprocessor`` class is updated.

  * Added new transformation methods

  * The executed transformation pipeline can now be saved and loaded from a json file. 
    This can be used to store and load the entire experimental data processing pipeline

  * The data handling methods of the class have changed some parameters to match the changes in the classes used

* The ``IQRPreprocessor`` class now is available in ``ambrosia.preprocessing``.

  * It can be used to remove outliers based on quartile and interquartile range estimates

* The ``RobustPreprocessor`` class is updated.

  * It now supports different types of tails for removal: ``both``, ``right`` or ``left``

  * For each processed column, a separate alpha portion of the distribution can be passed.

* The ``BoxCoxTransformer`` class now is available in ``ambrosia.preprocessing``

  * It can be used for data distribution normalization.

* The ``LogTransformer`` class now is available in ``ambrosia.preprocessing``

  * It can be used to transform data for variance reduction.

* The ``MLVarianceReducer`` class is updated.

  * Now it can store and load the selected ML model from a single specified path

Version 0.2.0 (22.11.2022)
---------------------------

Library name changed back to ``ambrosia``. Naming conflict in PyPI has been resolved.  
0.1.x versions are still available in PyPI under ``ambrozia`` name.

Version 0.1.2 (16.11.2022)
---------------------------

Hotfix for Ttest stat criterion absolute effect calculation. 
Url to main image deleted from docs.

Version 0.1.1 (04.10.2022)
---------------------------

Hotfix for library naming. 
Library temprorary renamed to ``ambrozia`` in PyPI repository due to hidden naming conflict. 

Version 0.1.0 (03.10.2022)
---------------------------

First release of ``Ambrosia`` package:

    * Added ``Designer`` class for experiment parameters design
    * Added ``Spliiter`` class for A/B groups split
    * Added ``Tester`` class for experiment effect measurement 
    * Added various classes for experiment data preprocessing
    * Added A/B testing tools with wide functionality  
