# Gender Pay Gap – MLOps Starter (Python)

This repo predicts **log(total compensation)** and quantifies an adjusted pay gap using a simple, testable ML pipeline.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
make install

# Put your data as CSV here:
#   data/data.csv
# with columns: jobtitle, gender, age, perfeval, education, dept, seniority, basepay, bonus

make test
make train
```

Artifacts (model + metrics) go to `artifacts/`.

## Data prep
If your source is an Excel file, export a CSV with those 9 columns. The training script computes:
- total_comp = basepay + bonus
- log_total_comp = log(total_comp)

## CI
A starter workflow is in `.github/workflows/python-ci.yml`.
On push/PR, it installs deps and runs tests.
