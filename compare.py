"""
Example 1: compare learning rates  ->  compare_learning_rates.png (gradient descent)
    for lr in 0.01 0.0314 0.1; do
        python3 train.py --patience 10  --no_plot --learning_rate $lr \
            --out_history history_lr$lr.csv --out_model model_lr$lr.npz
    done

    python3 compare.py history_lr0.01.csv history_lr0.0314.csv history_lr0.1.csv \
        --labels "lr 0.01" "lr 0.0314" "lr 0.1" \
        --out compare_learning_rates.png

Example 2: Compare Adam learning rates -> compare_adam_lr.png (Adam)

    for lr in 0.0001 0.001 0.003 0.01 0.03; do
        python3 train.py \
            --optimizer adam \
            --no_plot \
            --learning_rate $lr \
            --patience 10 \
            --out_history history_adam_$lr.csv \
            --out_model model_adam_$lr.npz
    done

    python3 compare.py \
        history_adam_0.0001.csv \
        history_adam_0.001.csv \
        history_adam_0.003.csv \
        history_adam_0.01.csv \
        history_adam_0.03.csv \
        --labels "Adam 0.0001" "Adam 0.001" \
                 "Adam 0.003" "Adam 0.01" "Adam 0.03" \
        --out compare_adam_lr.png

Example 3: Compare Adam vs Gradient Descent -> compare_optimizers.png

    # Gradient Descent (learning rate: 0.03)
    python3 train.py \
        --optimizer gradient_descent \
        --no_plot \
        --patience 10 \
        --out_history history_gd.csv \
        --out_model model_gd.npz

    # Adam (learning rate: 0.01)
    python3 train.py \
        --optimizer adam \
        --no_plot \
        --patience 10 \
        --out_history history_adam.csv \
        --out_model model_adam.npz

    # Compare both optimizers
    python3 compare.py \
        history_gd.csv history_adam.csv \
        --labels "Gradient Descent (default)" "Adam (0.01)" \
        --out compare_optimizers.png

"""

import argparse
import csv
import os

import matplotlib.pyplot as plt


def read_history(path):
    history = {}
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        for name in reader.fieldnames:
            history[name] = []
        for row in reader:
            for name in reader.fieldnames:
                history[name].append(float(row[name]))
    return history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("histories", nargs="+")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--out", type=str, default="compare_curves.png")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in args.histories]
    else:
        labels = args.labels
    if len(labels) != len(args.histories):
        raise ValueError("give exactly one label per history file")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    for path, label in zip(args.histories, labels):
        history = read_history(path)
        epochs = history["epoch"]
        ax_loss.plot(epochs, history["val_loss"], label=label)
        ax_acc.plot(epochs, history["val_acc"], label=label)

    ax_loss.set_xlabel("epochs")
    ax_loss.set_ylabel("validation loss")
    ax_loss.set_title("Validation loss")
    ax_loss.legend()

    ax_acc.set_xlabel("epochs")
    ax_acc.set_ylabel("validation accuracy")
    ax_acc.set_title("Validation accuracy")
    ax_acc.legend()

    plt.tight_layout()
    plt.savefig(args.out)
    print(f"> saved '{args.out}'")


if __name__ == "__main__":
    main()