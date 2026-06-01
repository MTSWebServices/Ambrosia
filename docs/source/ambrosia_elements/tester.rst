==================
Effect Measurement
==================

Tools for assessing the statistical significance of completed experiments
and calculating the experimental uplift value with corresponding confidence intervals.

.. admonition:: Multiple testing correction
   :class: caution

   When several hypotheses (number of variant combinations * number of metrics passed) are tested,
   the groups are compared in pairs and the p-values are adjusted for multiplicity. The
   ``correction_method`` parameter selects the procedure: ``"bonferroni"`` (default), ``"sidak"``,
   ``"holm"``, ``"holm-sidak"``, ``"fdr_bh"`` (Benjamini-Hochberg), ``"fdr_by"`` (Benjamini-Yekutieli),
   ``"hommel"`` or ``"simes-hochberg"`` (pass ``None`` to disable). The Benjamini-Hochberg and
   Benjamini-Yekutieli procedures control the false discovery rate; the others control the
   family-wise error rate. For ``"bonferroni"`` and ``"sidak"`` the confidence intervals are widened
   accordingly; the step-wise methods adjust only the p-values and leave the intervals at the nominal level.
   Hypotheses whose p-value cannot be computed still count toward the family size.


.. currentmodule:: ambrosia.tester

.. autosummary::
    :nosignatures:

    Tester
    test

----

.. autoclass:: Tester
   :members: run
.. autofunction:: test

Examples of using testing tools
-------------------------------

.. toctree::
    :maxdepth: 1

    /pandas_examples/06_pandas_tester
    /spark_examples/09_spark_tester