# Release benchmark

`release.json` is consumed by the shared MindIE release benchmark runner.
Each repetition executes `release.py` against the current checkout.

One measured operation performs two real `select_runtimes` calls on a synthetic
64-runtime catalog: it selects two matching A3 runtimes on distinct hosts, then
checks that reducing available capacity to one host refuses the entire group.
Every iteration checks both results. Inputs contain no user or machine data.

The metric is elapsed microseconds per pair of placement decisions, including
the assertions and request construction. It is a local CPU microbenchmark,
not end-to-end NPU throughput or a measured agent productivity improvement.
Compare runs only with their recorded source, Python, platform, and CPU context.
