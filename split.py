
import argparse
import numpy as np
import pandas as pd


COLUMN_NAMES = (
    ["id", "diagnosis"] +
    [f"feature_{i}" for i in range(30)]
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data.csv")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_train", type=str, default="data_train.csv")
    parser.add_argument("--out_valid", type=str, default="data_valid.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.dataset, header=None, names=COLUMN_NAMES)

    rng = np.random.default_rng(args.seed)
    shuffled_indices = rng.permutation(len(df))
    df_shuffled = df.iloc[shuffled_indices].reset_index(drop=True)
    n_train = int(len(df_shuffled) * args.train_ratio)
    first_part = range(0, n_train)
    rest_part = range(n_train, len(df_shuffled))

    df_train = df_shuffled.iloc[first_part]
    df_valid = df_shuffled.iloc[rest_part]
    df_train.to_csv(args.out_train, index=False, header=False)
    df_valid.to_csv(args.out_valid, index=False, header=False)

    print(f"total rows      : {len(df_shuffled)}")
    print(f"train rows      : {len(df_train)} -> {args.out_train}")
    print(f"valid rows      : {len(df_valid)} -> {args.out_valid}")
    print(f"train balance   : {df_train['diagnosis'].value_counts().to_dict()}")
    print(f"valid balance   : {df_valid['diagnosis'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()