import argparse

import matplotlib.pyplot as plt

from network import Network
from preprocessing import load_dataset, get_features, encode_labels, fit_scaler, apply_scaler, save_scaler, save_history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data_train.csv")
    parser.add_argument("--valid", type=str, default="data_valid.csv")
    parser.add_argument("--layer", type=int, nargs="+", default=[24, 24])
    parser.add_argument("--loss", type=str, default="categoricalCrossentropy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.0314)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_model", type=str, default="saved_model.npz")
    parser.add_argument("--out_scaler", type=str, default="saved_scaler.npz")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--out_history", type=str, default="history.csv")
    parser.add_argument("--optimizer", type=str, default="gradient_descent", choices=["gradient_descent", "adam"])
    return parser.parse_args()


def plot_curves(history, out_path="learning_curves.png", show=True):
    epochs_range = range(1, len(history["loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs_range, history["loss"], label="training loss")
    ax_loss.plot(epochs_range, history["val_loss"], "--", label="validation loss")
    ax_loss.set_xlabel("epochs")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs_range, history["acc"], label="training acc")
    ax_acc.plot(epochs_range, history["val_acc"], "--", label="validation acc")
    ax_acc.set_xlabel("epochs")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Learning Curves")
    ax_acc.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    if show:
        plt.show()


def main():
    args = parse_args()

    if len(args.layer) < 2:
        raise ValueError("At least two hidden layers are required.")

    df_train = load_dataset(args.train)
    df_valid = load_dataset(args.valid)

    X_train_raw = get_features(df_train)
    X_valid_raw = get_features(df_valid)

    y_train = encode_labels(df_train)
    y_valid = encode_labels(df_valid)

    mean, std = fit_scaler(X_train_raw)
    X_train = apply_scaler(X_train_raw, mean, std)
    X_valid = apply_scaler(X_valid_raw, mean, std)

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_valid.shape}")

    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]
    layer_sizes = [n_features] + args.layer + [n_classes]
    activations = ["sigmoid"] * len(args.layer) + ["softmax"]
    print(f"layer sizes : {layer_sizes}")
    print(f"activations : {activations}")
    print(f"optimizer   : {args.optimizer}")

    net = Network(layer_sizes, activations, seed=args.seed)

    history = net.fit(
        X_train, y_train, X_valid, y_valid,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        verbose=True,
        patience=args.patience,
        optimizer=args.optimizer,
    )

    net.save(args.out_model)
    save_scaler(args.out_scaler, mean, std)
    save_history(history, args.out_history)
    print(f"> saving model '{args.out_model}' to disk...")
    print(f"> saving history '{args.out_history}' to disk...")

    plot_curves(history, show=not args.no_plot)


if __name__ == "__main__":
    main()