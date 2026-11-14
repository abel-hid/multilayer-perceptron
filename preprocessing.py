
import numpy as np
import pandas as pd
import csv

COLUMN_NAMES = (
    ["id", "diagnosis"] +
    [f"feature_{i}" for i in range(30)]
)
FEATURE_COLUMNS = COLUMN_NAMES[2:]


def load_dataset(path):
    return pd.read_csv(path, header=None, names=COLUMN_NAMES)


def encode_labels(df):
    labels = []
    for diagnosis in df["diagnosis"]:
        if diagnosis == "B":
            labels.append([1, 0])   # benign
        elif diagnosis == "M":
            labels.append([0, 1])   # malignant
        else:
            raise ValueError(f"unexpected diagnosis value: {diagnosis}")
    return np.array(labels, dtype=float)

def get_features(df):
    return df[FEATURE_COLUMNS].to_numpy(dtype=float)


def fit_scaler(X_train):

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_scaler(X, mean, std):
    return (X - mean) / std


def save_scaler(path, mean, std):
    np.savez(path, mean=mean, std=std)


def load_scaler(path):
    data = np.load(path)
    return data["mean"], data["std"]

def save_history(history, path="history.csv"):
    metric_names = list(history.keys())
    n_epochs = len(history[metric_names[0]])

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch"] + metric_names)
        for i in range(n_epochs):
            row = [i + 1]
            for name in metric_names:
                row.append(round(float(history[name][i]), 6))
            writer.writerow(row)