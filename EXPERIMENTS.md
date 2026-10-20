# Experiments Log

Record only results you actually measured.

| Experiment | Change | Metric before | Metric after | Interpretation |
|---|---|---:|---:|---|
| Baseline comparison | Random Forest vs 7-day moving average | Fill after run | Fill after run | Which model performs better overall and by SKU? |
| Service level | Change service_z | N/A | N/A | How does the target service factor change safety stock and order quantities? |
| Review period | 7 vs 14 days | N/A | N/A | How does less frequent review change target stock? |

## Notes
- Run `python run_pipeline.py` after changing the experiment.
- Keep a copy of each metrics output if you want to compare experiments.
- Never claim a percentage cost reduction without a cost model and a measured comparison.
