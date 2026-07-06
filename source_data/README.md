# Source Data and Processing Scripts

This directory contains the source data and processing scripts used for the
Pepti-Agent manuscript figures and source-data tables.

## Directory layout

- `origin/`: original peptide records for the soluble, non-fouling, and hemolysis source pools.
- `refined/`: matched refined peptide records for the same three source pools.
- `results/`: scripts for pooled 300-peptide refinement analyses and figure source-data tables.
- `nonagent/`: matched 24-start fixed-length search and aggressive-refinement source data, plus processing scripts.

## Reproduction entry points

Run these commands from the repository root:

```bash
python source_data/results/make_peptide_refinement_figure_v6.py
python source_data/nonagent/generate_variants.py
python source_data/nonagent/make_agent_nonagent_comparison.py
python source_data/nonagent/make_aggressive_refinement_figure.py
python source_data/nonagent/make_nonagent_review_revisions_v3.py
```

The scripts write derived CSV source-data tables and figure artifacts under
`source_data/results/result/`, `source_data/nonagent/`, and
`source_data/nonagent/review_revisions/`.

## Manuscript mapping

- Pooled 300-peptide refinement data: `source_data/origin/`, `source_data/refined/`, and generated outputs from `source_data/results/make_peptide_refinement_figure_v6.py`.
- Fixed-length exhaustive-search analysis: `source_data/nonagent/selected_peptides_for_optimizer_comparison.csv`, `source_data/nonagent/generate_variants.py`, and `source_data/nonagent/nonagent_greedy_summary.csv`.
- Aggressive-refinement analysis: `source_data/nonagent/aggressive_refinement_summary.csv` and generated outputs from `source_data/nonagent/make_aggressive_refinement_figure.py`.
- Figure source data: generated CSV files from the scripts listed above.

All values are predictor-derived computational outputs; no wet-lab assay data
are included.
