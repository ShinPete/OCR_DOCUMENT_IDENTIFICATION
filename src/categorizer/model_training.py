from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score

ROOT = ROOT = Path.cwd().parents[1]  # if running from notebooks/, ROOT.parents[0] is repo root. Adjust if needed.
REPO = ROOT if (ROOT / "data").exists() else ROOT.parents[0]

def main():
    # 1) load splits
    train = pd.read_csv(REPO / "data/processed/datasets/v1/train.csv")
    val   = pd.read_csv(REPO / "data/processed/datasets/v1/validation.csv")
    # test = pd.read_csv(REPO / "data/processed/datasets/v1/test.csv")  # optional now

    # 2) labels
    le = LabelEncoder()
    y_train = le.fit_transform(train["type"])
    y_val   = le.transform(val["type"])

    # 3) model pipeline (word bi-grams is fine to start. try char 3–5 next)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, min_df=2, sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=None, class_weight="balanced", multi_class="auto"))
    ])

    # 4) train + eval
    pipe.fit(train["text"].astype(str), y_train)
    y_hat = pipe.predict(val["text"].astype(str))

    print(classification_report(y_val, y_hat, target_names=le.classes_, digits=4))
    print("Macro-F1:", f1_score(y_val, y_hat, average="macro"))

if __name__ == "__main__":
    main()
