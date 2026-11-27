# Results - Smart Spray Helper (tomato)

| Item | Value |
|---|---|
| Validation accuracy | 0.8125 (81.25%) |
| Field accuracy (PlantDoc) | 0.2016 (150/744) |
| Drop (lab -> field) | 61.1 points |
| Field photos used | 744 |
| Severity examples | see results/severity_summary.txt |
| Mode A (litres/s) | 4.5 L / 225 s |
| Mode B (litres/s) | 2.5 L / 195 s |
| Pesticide saved | 44.4% |
| Time saved | 13.3% |
| RAG bm25 hit rate | 1.000 (15/15) |
| RAG dense hit rate | 0.733 (11/15) |
| RAG hybrid hit rate | 0.867 (13/15) |
| Docs | 10 |
| Questions | 15 |

## Assumptions
- CONF_MIN = 0.6 (below -> retake)
- HEAT_THRESH = 0.5
- IMG_SIZE = 160, MAX_PER_CLASS = 200 (CPU only)
- SEED = 42, BATCH = 32
- Severity cut-offs 12% / 50% (Horsfall-Barratt simplified)
- field_sim DOSE_FACTOR, sec_move 10s, sec_spray 15s, litres_per_cell 0.5 (assumptions)
- Weather rules from US/Canadian extension guidance (no Indian equivalent found)
- Severity colour-based, works best on plain-background single-leaf photos

## Not completed
- Deployment (optional, not done)
- Accuracy below 85% target (81.25%); one change attempted (more epochs).
