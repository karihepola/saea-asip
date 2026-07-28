Scripts for running results for the Surrogate-Assisted Optimization of Application-Specific Processors paper

[Zenodo dataset](https://zenodo.org/records/21649988)


Download datasets:
```bash
./scripts/init.sh
```

- `cycle_count_estimation/` — contains the code running cycle count estimation.
- `synthesis_estimation/` — contains the code and data required for running synthesis estimation.

To process the final optimization results:
```bash
python3 scripts/print_opt_results.py
```