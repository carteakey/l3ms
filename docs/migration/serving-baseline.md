# Serving source reconciliation — 2026-10-07

The migration branch now preserves the committed `llama-swap.yaml` from the
original L3MS checkout at `ed7538a`, including the platinum Strata profiles,
updated binaries/quantization, and the 32k vision context. This is a source sync,
not a live reload or a successful coordinated-engine acceptance test. The
original checkout, its uncommitted changes, and running services remain untouched.

Selected Qwen3.8 dashboard metadata comes from the same committed source.
Throughput and hardware claims are source-reported historical measurements;
the migration has not remeasured them. Other historical dashboard results retain
their earlier metadata and may have different hardware settings.

The source metadata references `bench-models/bench-qwen38-refresh-ab.sh` and
`bench-models/bench_strata_long_horizon.py`, neither present in that committed
source tree. They have not been copied from uncommitted work or represented as
available benchmark commands. The dashboard retains their names as
`originalBenchmark`, links here, and leaves the executable benchmark command
unrecorded. Importing and validating those harnesses is a remaining parity gate.

A read-only review of the currently uncommitted harnesses confirmed why they
must not be imported verbatim: the refresh driver uses a fixed shared temporary
directory, an original-checkout absolute configuration path and broad
process-name cleanup; the long-horizon driver embeds an authentication token.
No value is reproduced here and neither harness was executed or copied. Their
source remains intact in the original checkout. A future import must first
replace those assumptions with explicit configuration, exact owned-process
cleanup and secret-free credential handling, then pass offline safety fixtures.

Before any live pilot, recheck current service state, binary flags, model paths,
GPU ownership and the exact rollback configuration. Imported metadata does not
prove installation, native health, ingress isolation or GPU release.
