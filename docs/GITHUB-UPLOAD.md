# GitHub Upload

Created by School of AI and School of QC.

Create an empty GitHub repository, extract this project, and run:

~~~bash
git init
git add .
git commit -m "Initial quantum reservoir computing system"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
git push -u origin main
~~~

Before pushing, run `ruff check .`, `ruff format --check .`, `pytest`, and `python -m build`.
Review `git status` and never commit credentials or private sequences. Generated CSVs,
timestamped experiment runs, caches, and build outputs are ignored. GitHub Actions repeats the
quality checks on Python 3.11 and 3.12 after upload.

The compact `examples/reference-run` evidence and selected `docs/assets` plots are intentionally
tracked. They make README claims auditable without checking in fitted joblib files or full
feature matrices.
