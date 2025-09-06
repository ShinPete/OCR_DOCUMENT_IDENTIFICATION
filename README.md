# OCR Document Identification — from raw files to labels

## Problem
Identify document type from PDFs/images to route workflows (intake, billing, etc).

## Data (sample)
`data_sample/` has 10 redacted docs across 3 classes + `labels.csv` (path,label).
No PII; real projects should replace with their own data.

## Method
Pipeline: OCR → clean → vectorize → classify.
Baselines: TF-IDF + linear (LogReg/SVM). 
Optional: small transformer (DistilBERT) for comparison.
Backtesting: 5-fold stratified, fixed seed.

| Model              | Accuracy | Macro F1 |
| ------------------ | -------: | -------: |
| TF-IDF + Logistic  |     0.92 | **0.91** |
| TF-IDF + LinearSVM |     0.90 |     0.89 |

Repo structure:
src/            # train.py, predict.py, eval.py, ocr_utils.py, text_clean.py
notebooks/      # 01_explore.ipynb, 02_modeling.ipynb
data_sample/    # tiny demo files + labels.csv (no PII)
models/         # vectorizer.pkl, classifier.pkl, meta.json (after training)
reports/        # figures/, metrics.json
config.yaml     # paths + model params
requirements.txt

Limits & next steps

OCR quality drives ceiling; add language detection & layout features; consider small transformer for hard classes.

## Quickstart
```bash
# 1) Setup
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2) Demo run on sample data
python -m src.train --config config.yaml
python -m src.predict --input data_sample --models_dir models --out out/preds.csv
python -m src.eval --gold data_sample/labels.csv --pred out/preds.csv --out reports/metrics.json
