
## Stats Analyzer (CLI)
`stats-analyzer` ingests tabular data and executes:
- Data health checks.
- Categorical validation (rule-based and/or joint-model based).
- Covariate validation (univariate and/or joint-model based).
- Identify-variable checks.
- ANOVA or ANCOVA modeling (via `statsmodels`).
- Adjusted means generation.
- Assumption diagnostics and flags.
- Figure generation.
- Table artifact generation (default + custom table specs, with display renaming and formatting).
- Optional PDF report generation (via `reportlab`).

### Package Layout
- Source code: `src/stats_analyzer/`
- CLI entrypoint: `src/stats_analyzer/cli/main.py`
- Report/table templates:
  - `templates/default_report_template.yaml`
  - `templates/table_config_example.yaml`
  - `templates/table_config_overflow_appendix.yaml` (auto overflow handling + wide tables appendix)
  - `templates/template_sdk_reference.yaml` (SDK-style dictionary of supported template/report/table/figure inputs)

### Installation
If you are using this package standalone:

```bash
pip install -e .
```

Or run directly from source:

```bash
# Windows PowerShell
$env:PYTHONPATH='src'
python -m stats_analyzer.cli.main --help
```

Core dependencies:
- `pandas`
- `numpy`
- `statsmodels`
- `matplotlib`
- `PyYAML`
- `reportlab` (required for PDF generation)

### Supported Input Data
`DataLoader` supports:
- `.csv`
- `.parquet` / `.pq`
- `.xlsx` / `.xls`

### CLI Commands
Use `--help` on any command. Use `--explain` for method/assumption context.

```bash
stats-analyzer ingest <input>
stats-analyzer check-health <input> ...
stats-analyzer validate-categorical <input> ...
stats-analyzer validate-covariates <input> ...
stats-analyzer validate-id-vars <input> ...
stats-analyzer run-model <input> ...
stats-analyzer plot <input> ...
stats-analyzer report <input> ...
stats-analyzer run-all <input> ...
```

### Common Arguments
- `--response`: response variable.
- `--primary-factor`: primary categorical factor.
- `--covariates`: ANCOVA covariates.
- `--group-vars`: grouping/stratification variables.
- `--id-vars`: candidate identify variables.
- `--analysis-type {auto,anova,ancova}`: auto chooses ANCOVA when covariates are provided, otherwise ANOVA.
- `--template`: report template path.
- `--table-config`: table specification template path (defaults to `--template` when omitted).
- `--no-tables`: disable table artifact generation.

Threshold and model options:
- `--alpha` (default `0.05`): general significance threshold.
- `--validation-alpha` (defaults to `--alpha`): validation-model threshold.
- `--assumption-alpha` (defaults to `--alpha`): model assumption-check threshold.
- `--min-total-n` (default `10`): minimum overall sample size.
- `--min-n-per-factor` (default `2`): minimum per primary factor level.
- `--min-n-per-group` (default `3`): minimum per group stratum from `--group-vars`.
- `--cov-type` (default `HC3`): ANCOVA covariance estimator (`HC3`, `HC1`, `nonrobust`, etc.).
- `--validation-mode {univariate,joint}`: covariate validation style.
- `--categorical-validation-mode {rules,joint,both}`: categorical validation style.
- `--assumption-scope {global,group,both}`: run assumption diagnostics on full model residuals, grouped residuals, or both.

### Methods Used
#### ANOVA
- Built with `statsmodels` OLS formula model.
- Formula includes primary factor and group variables plus factor-by-group interactions.
- Type II ANOVA table is computed.

#### ANCOVA
- Built with `statsmodels` OLS formula model with:
  - `C(primary_factor)`
  - covariates
  - `C(group_variables)`
- Default robust covariance: `HC3`.
- Slope homogeneity screen uses interaction terms `C(primary_factor):covariate` and flags violations.

#### Adjusted Means
- ANOVA: group means by primary factor.
- ANCOVA: predictions at covariate means (with group variables anchored to modal values).
- Extrapolation flags are emitted when adjusted-mean rows are outside observed covariate ranges or use unobserved categorical strata.

### Validation Methods
#### Data Health
Checks:
- Required columns present.
- Missingness per required variable.
- Total N threshold.
- Group stratum N threshold (`--min-n-per-group`).

#### Categorical Validation
Modes:
- `rules`: level count and minimum-N thresholds (including group-factor strata).
- `joint`: single joint categorical model checks.
- `both`: executes both.

Joint categorical validation can flag:
- Rank deficiency.
- Non-positive residual df.
- High condition number.
- Weak categorical term signals.
- Residual non-normality.
- Heteroscedasticity.
- Autocorrelation signal.

#### Covariate Validation
Always checks each covariate for:
- Presence.
- Numeric castability.
- Zero variance.
- Missingness.

`--validation-mode joint` adds a joint model and can flag:
- Weak covariate signals.
- Residual non-normality.
- Heteroscedasticity.
- Linearity misspecification (RESET).
- Autocorrelation signal.

Covariate validation metrics now include per-covariate profile fields
(presence, numeric castability, missingness, variance, summary stats) and,
in `joint` mode, joint-model diagnostics (formula, design rank/columns,
condition number, covariate p-values, and assumption test statistics).

#### Identify Variable Validation
Checks:
- Presence of ID variables.
- Duplicate counts (uniqueness warnings).

### Assumption Diagnostics (Model Stage)
Assumption checks run on fitted model output and are reported in metrics/flags:
- Residual normality: Jarque-Bera.
- Homoscedasticity: Breusch-Pagan.
- Linearity (when covariates exist): Ramsey RESET.
- Independence signal: Durbin-Watson.
- Influence: Cook's distance threshold count.
- Collinearity/stability signal: condition number.
- ANCOVA slope homogeneity: propagated from interaction screen.

When `--assumption-scope group` (or `both`) is used, additional residual diagnostics are emitted per group
using `primary_factor + group_vars` strata.

Current policy:
- Most assumption violations are warnings.
- Slope non-homogeneity is flagged as error.

### Outputs
For `report` / `run-all`:
- Figures: `output_dir/figures/*.png`
- Tables: `output_dir/tables/*.csv`
- PDF report: `output_dir/analysis_report.pdf` (when `reportlab` is available)

Run summary includes JSON fields like:
- `analysis_type`
- `figure_count`
- `table_count`
- `flag_count`
- `report_path`

### Table Output Configuration
Table generation is template-driven from `tables:` config.

Default table sources include:
- `flags`
- `fit_statistics`
- `parameter_estimates`
- `adjusted_means`
- `assumption_metrics`
- `validation_summary`
- `dataset`
- `figures`

Custom tables can define:
- `id`
- `source`
- `query`
- `columns`
- `rename` (display headers independent from internal variable names)
- `sort_by`, `ascending`
- `limit`
- `include_in_pdf`
- `format` (column formatting)

Useful custom `source` values for diagnostics:
- `covariate_metrics` (flattened rows from covariate validation metrics)
- `categorical_metrics`
- `health_metrics`
- `identify_variable_metrics`
- `validation_metrics` (all validation metrics with a `validation` column)

PDF table overflow controls (`tables:`):
- `pdf_overflow_mode`: `auto` (default), `portrait`, `landscape`, or `split`.
- `pdf_overflow_placement`: `inline` (default) or `appendix` for overflowed tables.
- `pdf_truncate_chars`: max characters per cell before PDF truncation.
- `pdf_min_col_width` / `pdf_min_col_width_truncated`: width budgeting floors.
- `pdf_max_col_width`: cap for wide columns before proportional shrinking.
- `pdf_split_key_columns`: number of left-most columns repeated across split parts.
- `pdf_split_orientation`: `auto`, `portrait`, or `landscape` for split pages.

Supported formatting:
- String short form: `"decimal:2"`, `"percent:1"`, `"integer"`, `"scientific:3"`
- Dict form with options:
  - `type`
  - `decimals`
  - `thousands`
  - `multiply_by_100` (for percent)
  - `prefix`, `suffix`
  - `null`

Example:

```yaml
tables:
  include_defaults: true
  default_formats:
    fit_statistics:
      value: "decimal:5"
  custom:
    - id: "high_outcome_subjects"
      source: "dataset"
      columns: ["subject_id", "treatment", "outcome", "rate"]
      rename:
        subject_id: "Subject ID"
        treatment: "Treatment Arm"
        outcome: "Observed Outcome"
        rate: "Response Rate"
      format:
        Observed Outcome: "decimal:2"
        Response Rate:
          type: percent
          decimals: 1
      query: "outcome >= 85"
      sort_by: ["outcome"]
      ascending: false
      limit: 25
      include_in_pdf: true
```

### Report Template Notes
- Template defaults are defined in `src/stats_analyzer/reporting/template_engine.py`.
- `tables.pdf_max_rows` and `tables.pdf_max_columns` cap preview size embedded in PDF.
- Full table data remains available in CSV files.

### Example End-to-End Invocation
```bash
# PowerShell
$env:PYTHONPATH='src'
python -m stats_analyzer.cli.main run-all sample_data/ancova_sample.csv `
  --response outcome `
  --primary-factor treatment `
  --covariates age baseline `
  --group-vars site `
  --id-vars subject_id `
  --template templates/default_report_template.yaml `
  --table-config templates/table_config_example.yaml `
  --output-dir output/stats_run
```

### Testing
Stats analyzer tests:

```bash
python -m pytest tests/stats_analyzer -q
```

### Known Runtime Constraint
If `reportlab` is unavailable, PDF creation is skipped and flagged, while figures/tables are still generated.
