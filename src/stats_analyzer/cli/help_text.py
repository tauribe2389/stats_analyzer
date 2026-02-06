EXPLANATIONS = {
    "ingest": (
        "Ingest reads a dataset and reports basic dimensions. "
        "Assumes a tabular file format: CSV, Parquet, or Excel."
    ),
    "check-health": (
        "Health checks validate required columns, missingness, and minimum sample footprint. "
        "Assumes variables passed to CLI map to dataset columns. "
        "Thresholds are configurable with --min-total-n, --min-n-per-factor, and --min-n-per-group."
    ),
    "validate-categorical": (
        "Categorical validation checks threshold rules and can run a fully joint categorical model. "
        "Use --categorical-validation-mode to choose rules, joint, or both."
    ),
    "validate-covariates": (
        "Covariate validation confirms covariates exist, are numeric, and show variation. "
        "Assumes covariates are suitable for linear adjustment in ANCOVA. "
        "Use --validation-mode joint for combined covariate screening."
    ),
    "validate-id-vars": (
        "Identify-variable validation checks existence and duplicate rates. "
        "Assumes ID uniqueness should hold unless duplicates are expected by design."
    ),
    "run-model": (
        "Model step fits ANOVA or ANCOVA using statsmodels. "
        "ANCOVA defaults to robust covariance HC3 (override with --cov-type). "
        "Assumption diagnostics are run and flagged (normality, homoscedasticity, independence, "
        "linearity, and slope homogeneity where applicable)."
    ),
    "plot": (
        "Plot step creates diagnostic and observed-vs-model comparison figures. "
        "Assumes a fitted model is available."
    ),
    "report": (
        "Report step generates a PDF using ReportLab and a merged template. "
        "Table artifacts are generated from defaults and optional custom table specs, "
        "including header renaming and numeric formatting rules."
    ),
    "run-all": (
        "Run-all executes ingest, checks, validation, modeling, plotting, and report generation. "
        "Tables and figures are both emitted for report use."
    ),
}
