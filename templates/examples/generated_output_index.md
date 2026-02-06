# Generated Example Output Index
The following sample runs were executed in this workspace using the example templates.

## 1) Clinical ANCOVA (successful model run)
- Profile: `templates/examples/analysis_profile_clinical.yaml`
- Input: `sample_data/examples/clinical_trial_ancova.csv`
- Output root: `output/example_runs/clinical_trial_ancova_v2`
- Summary: `output/example_runs/clinical_trial_ancova_v2/run_summary.json`
- Figures:
  - `output/example_runs/clinical_trial_ancova_v2/figures/diagnostic_residuals.png`
  - `output/example_runs/clinical_trial_ancova_v2/figures/comparison_observed_vs_model.png`
- Tables:
  - `output/example_runs/clinical_trial_ancova_v2/tables/fit_statistics.csv`
  - `output/example_runs/clinical_trial_ancova_v2/tables/adjusted_means.csv`
  - `output/example_runs/clinical_trial_ancova_v2/tables/assumption_metrics.csv`
  - `output/example_runs/clinical_trial_ancova_v2/tables/validation_status.csv`
  - `output/example_runs/clinical_trial_ancova_v2/tables/dataset_missingness_focus.csv`

## 2) Manufacturing ANOVA (successful model run)
- Profile: `templates/examples/analysis_profile_operations.yaml`
- Input: `sample_data/examples/manufacturing_anova.csv`
- Output root: `output/example_runs/manufacturing_anova_v3`
- Summary: `output/example_runs/manufacturing_anova_v3/run_summary.json`
- Figures:
  - `output/example_runs/manufacturing_anova_v3/figures/diagnostic_residuals.png`
  - `output/example_runs/manufacturing_anova_v3/figures/comparison_observed_vs_model.png`
- Tables:
  - `output/example_runs/manufacturing_anova_v3/tables/fit_statistics.csv`
  - `output/example_runs/manufacturing_anova_v3/tables/adjusted_means.csv`
  - `output/example_runs/manufacturing_anova_v3/tables/parameter_estimates.csv`
  - `output/example_runs/manufacturing_anova_v3/tables/validation_rollup.csv`
  - `output/example_runs/manufacturing_anova_v3/tables/night_shift_batches.csv`
  - `output/example_runs/manufacturing_anova_v3/tables/flags.csv`

## 3) Sparse Group QC (intentional validation-fail demo)
- Profile: `templates/examples/analysis_profile_sparse_qc.yaml`
- Input: `sample_data/examples/sparse_groups_demo.csv`
- Output root: `output/example_runs/sparse_group_qc_v2`
- Summary: `output/example_runs/sparse_group_qc_v2/run_summary.json`
- Tables:
  - `output/example_runs/sparse_group_qc_v2/tables/flags.csv`
  - `output/example_runs/sparse_group_qc_v2/tables/flag_rollup.csv`

## Note on PDF
`report_path` is `null` in these runs because `reportlab` is not available in this environment.  
All non-PDF artifacts (tables, figures, summaries) were generated successfully.

