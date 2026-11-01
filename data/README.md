# Data

Datasets are NOT stored in git. Each teammate downloads them locally into `data/raw/`.
- Register every dataset in `datasets.yaml` (source, licence, real vs synthetic, label source).
- Put download/prepare steps in `scripts/` (one script per dataset).
- Processed files go to `data/processed/` (also not committed).
