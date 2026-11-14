

import argparse
import numpy as np

from preprocessing import load_dataset, encode_labels, get_features, apply_scaler, load_scaler
from network import Network, accuracy, confusion_counts, precision, recall, f1_score

def binary_crossentropy(true_malignant, predicted_malignant_prob):
    """
        E = -1/N * sum( y*log(p) + (1-y)*log(1-p) )
    """
    epsilon = 1e-12
    p = np.clip(predicted_malignant_prob, epsilon, 1 - epsilon)
    y = true_malignant
    per_example_loss = y * np.log(p) + (1 - y) * np.log(1 - p)
    return -np.mean(per_example_loss)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data_valid.csv")
    parser.add_argument("--model", type=str, default="saved_model.npz")
    parser.add_argument("--scaler", type=str, default="saved_scaler.npz")
    return parser.parse_args()


def main():
    args = parse_args()
    net = Network.load(args.model)
    mean, std = load_scaler(args.scaler)
    df = load_dataset(args.dataset)
    X_raw = get_features(df)
    y_true = encode_labels(df) 
    X = apply_scaler(X_raw, mean, std)
    predicted_probs = net.forward(X)
    acc = accuracy(y_true, predicted_probs)
    true_malignant = y_true[:, 1]
    predicted_malignant_prob = predicted_probs[:, 1]
    bce = binary_crossentropy(true_malignant, predicted_malignant_prob)



    tp, fp, tn, fn = confusion_counts(y_true, predicted_probs)
    precision_value = precision(tp, fp)
    recall_value = recall(tp, fn)
    f1_value = f1_score(precision_value, recall_value)
    print(f"dataset          : {args.dataset}")
    print(f"n_samples        : {X.shape[0]}")
    print(f"accuracy         : {acc:.4f}")
    print(f"binary cross-entropy error: {bce:.4f}")
    print(f"precision        : {precision_value:.4f}")
    print(f"recall           : {recall_value:.4f}")
    print(f"f1               : {f1_value:.4f}")
    print(f"confusion        : TP={tp} FP={fp} TN={tn} FN={fn}")


if __name__ == "__main__":
    main()