# Document Categorizer

Classifies documents by type—from raw files to labels. Uses OCR for images/PDFs, light text cleanup, TF-IDF vectorization, and a lightweight linear classifier.

## Pipeline

**OCR → clean → vectorize → train → predict → report**

## Methodology & Experiments

| Run ID | Features (TF-IDF)          | Vocab / Filtering                   | Classifier     | Key Params                     | Macro-F1 (Val) | Macro-F1 (Test) | Notes |
|:-----:|-----------------------------|-------------------------------------|----------------|-------------------------------|:--------------:|:---------------:|------|
| A1    | **word** n-grams (1,1)      | max_features=50k • min_df=2 • max_df=0.90 | LogisticReg    | C=1.0 • class_weight=balanced | 0.89x*         | 0.89x*          | Baseline word unigrams |
| A2    | **word** n-grams (1,2)      | max_features=100k • min_df=2 • max_df=0.90 | LogisticReg    | C=1.0 • class_weight=balanced | 0.89x*         | 0.89x*          | Little to no change vs (1,1) |
| B1    | **char** n-grams (3,5)      | max_features=100k • min_df=2 • max_df=0.90 | LogisticReg    | C=1.0 • class_weight=balanced | 0.90x*         | 0.91x*          | Robust to OCR noise |
| B2    | **char** n-grams (3,5)      | max_features=100k • min_df=2 • max_df=0.90 | **LinearSVC**  | C=1.0 • max_iter=5000         | **0.90–0.91*** | **0.93x***      | Current winner |
| C1    | DistilBERT (fine-tuned)     | –                                   | Transformer    | 3 epochs • max_len=256         | 0.93x*         | 0.93x*          | Ties LinearSVC in this data |
| D1    | B2 + C1 (simple blend)      | –                                   | Ensemble       | 0.5× SVC + 0.5× BERT probs     | 0.93x*         | 0.93x*          | No measurable lift here |

\* Replace with your actual scores from `reports/metrics_log.txt` / test runs.
*** 5-fold CV range: 0.90–0.91 (val), 0.92–0.93 (test)

---

## Quick start

```bash

# 1) Install
pip install -r requirements.txt

# 2) Run a tiny demo on sample docs (adjust --images_dir if needed)
python -m src.ocr.extract --images_dir data/raw --out_csv data/processed/ocr_output.csv

# 3) Train a baseline (char 3–5 + LinearSVC) using your split CSVs
python -m src.classify.train_baseline \
  --data_csv data/processed/datasets/v1/train.csv \
  --val_csv  data/processed/datasets/v1/validation.csv \
  --models_dir models \
  --analyzer char --ngram_min 3 --ngram_max 5 --model linearsvc

# 4) Evaluate on the test split and write a report
python -m src.classify.evaluate \
  --model_path models/<your_model>.joblib \
  --data_csv   data/processed/datasets/v1/test.csv \
  --reports_dir reports

```



# Data Layout

data/
  raw/                 # input files (images/PDFs)
  processed/
    ocr_output.csv     # canonical text table: filename,text,label
    datasets/
      v1/
        train.csv
        validation.csv
        test.csv
models/
reports/
assets/                # tracked images for README (e.g., confusion matrices)

# Training

python -m src.classify.train_baseline \
  --data_csv data/processed/datasets/v1/train.csv \
  --val_csv  data/processed/datasets/v1/validation.csv \
  --models_dir models \
  --analyzer char --ngram_min 3 --ngram_max 5 --model linearsvc \
  --max_features 100000 --max_df 0.9 --min_df 2
  
# Prediction / Evaluation
python -m src.classify.evaluate \
  --model_path models/<your_model>.joblib \
  --data_csv   data/processed/datasets/v1/test.csv \
  --reports_dir reports
 
![Confusion matrix](assets/cm_test_v2.png)
