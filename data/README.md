# Data

Created by School of AI and School of QC.

The default benchmark generates a deterministic Mackey–Glass sequence at runtime. Run
`python scripts/generate_series.py` to write `data/mackey_glass.csv`; generated CSV files are
ignored by Git.

For saved-model forecasting, provide a CSV with one finite numeric column named `value`. The file
must contain at least as many rows as the fitted autoregressive lag count, which is 20 in the
default configuration. Rows must be ordered from oldest to newest. Extra columns are ignored.

The included project contains no personal, proprietary, or production data. If you adapt it to
real series, document provenance and sampling, obtain permission, protect sensitive fields, and
avoid random train/test shuffling.
