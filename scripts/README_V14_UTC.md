# V14-UTC Script Entry Points

Formal documentation now lives in:

```text
maxcut/v14_utc/
```

Formal algorithm:

```bash
bash scripts/run_v14_four_scheme_random50_all.sh
```

Higher-utilization mode:

```bash
WORKERS_PER_GPU=2 bash scripts/run_v14_four_scheme_random50_all.sh
```

Merge shard results:

```bash
python scripts/merge_v14_four_scheme_seed_benchmark.py --output-dir outputs/v14_four_scheme_random50
```

Core scripts kept at top level are the formal V14-UTC runner and its import
dependencies. Selected old helper scripts are kept in:

```text
maxcut/v14_utc/scripts_archive/
```
