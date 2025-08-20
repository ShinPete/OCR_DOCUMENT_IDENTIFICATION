import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.ocr import util


def main():
    # Load the dataset
    df = pd.read_csv(r'C:\Users\abajp\PycharmProjects\BofAOCRProject\data\processed\output.csv')

    # Display the first few rows of the dataframe
    print(df.head())
    df = pd.read_csv(r'C:\Users\abajp\PycharmProjects\BofAOCRProject\data\processed\output.csv')
    nulls = df[df.isna().any(axis=1)]
    nulls.to_csv("null_rows_snapshot.csv", index=False)

    # Drop nulls & empty strings in core columns
    df = df.dropna(subset=["text", "type"])
    df = df[df["text"].str.strip() != ""]
    from sklearn.model_selection import train_test_split
    train_val, test = train_test_split(df, test_size=0.1, random_state=42, stratify=df['type'])
    train, val = train_test_split(train_val, test_size=0.09, random_state=42, stratify=train_val['type'])
    train_val_pct = train_val.value_counts('type') / len(train_val) * 100
    test_val_pct = test.value_counts('type') / len(test) * 100
    diff = train_val_pct - test_val_pct
    results = util.compare_distributions(df, [('train', train), ('val', val), ('test', test)], 'type')
    train.to_csv("../data/processed/datasets/v1/train.csv", index=False)
    val.to_csv("../data/processed/datasets/v1/validation.csv", index=False)
    test.to_csv("../data/processed/datasets/v1/test.csv", index=False)
    # %%



if __name__ == "__main__":
    main()
