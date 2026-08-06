"""Dataset loaders for the additional chemistry benchmarks.

These loaders ONLY prepare data. They download/parse a raw dataset, compute the
low-fidelity (LF) and high-fidelity (HF) targets, featurize the candidate pool to
a 10-D PCA embedding, cache the featurized pool, and write a numeric CSV that the
*existing* ``src/benchmark.py`` pipeline consumes unchanged via
``ChemistryBenchmark(..., use_smiles=False)``.

Nothing here touches the MFBO loop, the EI acquisition, fidelity selection, the
transfer-learning surrogates, the 20-seed protocol, or regret/budget tracking.
The featurized CSV is exactly the same kind of numeric ``f0..f9, HF, LF`` table
that ChemistryBenchmark already standardizes and runs on — so the new benchmarks
inherit the identical pipeline and evaluation protocol as FreeSolv / COFs.

Run a loader directly to (down)load + featurize + cache + print diagnostics::

    python -m benchmarks.hopv15
    python -m benchmarks.matbench_gap
"""
