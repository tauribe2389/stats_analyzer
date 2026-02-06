# Stats Analyzer Example Profiles
This folder contains reusable example analysis profiles and template pairs.

## Files
- `analysis_profile_clinical.yaml`
- `analysis_profile_operations.yaml`
- `analysis_profile_sparse_qc.yaml`
- `report_*_template.yaml`
- `table_*_template.yaml`

## Corresponding Example Data
- `sample_data/examples/clinical_trial_ancova.csv`
- `sample_data/examples/manufacturing_anova.csv`
- `sample_data/examples/sparse_groups_demo.csv`

## Manual Run Examples
PowerShell commands:

```powershell
$env:PYTHONPATH='src'

python -m stats_analyzer.cli.main run-all `
  sample_data/examples/clinical_trial_ancova.csv `
  --response outcome `
  --primary-factor treatment `
  --covariates age baseline `
  --group-vars region `
  --id-vars subject_id `
  --analysis-type ancova `
  --validation-mode joint `
  --categorical-validation-mode both `
  --template templates/examples/report_clinical_template.yaml `
  --table-config templates/examples/table_clinical_template.yaml `
  --output-dir output/example_runs/clinical_trial_ancova

python -m stats_analyzer.cli.main run-all `
  sample_data/examples/manufacturing_anova.csv `
  --response quality_score `
  --primary-factor process `
  --group-vars shift plant `
  --id-vars batch_id `
  --analysis-type anova `
  --min-n-per-factor 3 `
  --min-n-per-group 6 `
  --template templates/examples/report_operations_template.yaml `
  --table-config templates/examples/table_operations_template.yaml `
  --output-dir output/example_runs/manufacturing_anova

python -m stats_analyzer.cli.main run-all `
  sample_data/examples/sparse_groups_demo.csv `
  --response score `
  --primary-factor cohort `
  --group-vars site `
  --id-vars id `
  --analysis-type anova `
  --min-n-per-factor 6 `
  --min-n-per-group 8 `
  --template templates/examples/report_sparse_qc_template.yaml `
  --table-config templates/examples/table_sparse_qc_template.yaml `
  --output-dir output/example_runs/sparse_group_qc
```

Notes:
- The sparse QC profile is expected to trigger blocking validation flags.
- If `reportlab` is not installed, PDF output is skipped but tables/figures are still generated.
