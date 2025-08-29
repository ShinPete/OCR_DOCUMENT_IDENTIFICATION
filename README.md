Document Categorizer
Classifies documents by type from raw files to labels. Uses OCR for images and PDFs, basic text cleanup, vectorization, and a lightweight classifier.

Pipeline
OCR → clean → vectorize → train → predict → report

Quick start
# 1) Install
pip install -r requirements.txt

# 2) Run a tiny demo on sample docs
python scripts/ocr.py --input data/raw --out data/ocr
python scripts/predict.py --input data/ocr --out out/preds.csv

# 3) See results
head -n 5 out/preds.csv
Data Layout
Training
python scripts/train.py
--train data/processed/train.csv
--valid data/processed/valid.csv
--models_dir models

Saves vectorizer.pkl, classifier.pkl, and meta.json. Prints accuracy, f1, and a confusion matrix.

Prediction python scripts/predict.py
--input data/ocr
--models_dir models
--out out/preds.csv

Output columns: path, predicted_label, score.

Evaluation python scripts/eval.py
--gold data/processed/test.csv
--pred out/preds.csv

Reports accuracy, macro F1, per-class F1.

Model artifacts

models/vectorizer.pkl

models/classifier.pkl

models/meta.json with params and class order

If NumPy types break JSON, use a safe serializer:

''' import json, numpy as np def to_ser(x): if hasattr(x, "tolist"): return x.tolist() if isinstance(x, (np.integer, np.floating, np.bool_)): return x.item() return str(x) json.dump(meta, open("models/meta.json","w"), indent=2, default=to_ser) ''' Limits

OCR errors reduce accuracy on low-quality scans.

Current language: en only.

Small training set. Some classes under-represented.

Roadmap

Better OCR cleanup and language detection.

Swap in a stronger classifier or a small transformer.

Simple web viewer for drag-and-drop predict.

License

MIT unless noted otherwise.

