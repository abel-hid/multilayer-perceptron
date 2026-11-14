import numpy as np
from layers import DenseLayer


def categorical_crossentropy(true_labels, predicted_probs):
    """
    E = -mean( sum_over_classes( true * log(predicted) ) )
    """
    epsilon = 1e-12
    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    per_example_loss = -np.sum(true_labels * np.log(predicted_probs), axis=1)
    return np.mean(per_example_loss)


def accuracy(true_labels, predicted_probs):
    predicted_classes = np.argmax(predicted_probs, axis=1)
    true_classes = np.argmax(true_labels, axis=1)
    return np.mean(predicted_classes == true_classes)

def confusion_counts(true_labels, predicted_probs):
    predicted_classes = np.argmax(predicted_probs, axis=1)
    true_classes = np.argmax(true_labels, axis=1)

    tp = 0 # really M and predicted M
    fp = 0 # really B but predicted M
    tn = 0 # really B and predicted B
    fn = 0 # really M but predicted B

    for i in range(len(true_classes)):
        if true_classes[i] == 1 and predicted_classes[i] == 1:
            tp += 1
        elif true_classes[i] == 0 and predicted_classes[i] == 1:
            fp += 1
        elif true_classes[i] == 0 and predicted_classes[i] == 0:
            tn += 1
        elif true_classes[i] == 1 and predicted_classes[i] == 0:
            fn += 1
    return tp, fp, tn, fn

def precision(tp, fp):
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(tp, fn):
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


class Network:
    def __init__(self, layer_sizes, activations, seed=None):
        rng = np.random.default_rng(seed)
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                DenseLayer(
                    n_inputs=layer_sizes[i],
                    n_outputs=layer_sizes[i + 1],
                    activation=activations[i],
                    rng=rng,
                )
            )
        self.history = {"loss": [], "val_loss": [], "acc": [], "val_acc": [],
                        "val_precision": [], "val_recall": [], "val_f1": []}

    def forward(self, input_batch):
        output_batch = input_batch
        for layer in self.layers:
            output_batch = layer.forward(output_batch)
        return output_batch

    def backward(self, true_labels, predicted_probs):
        grad = predicted_probs - true_labels

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, learning_rate, optimizer="gradient_descent", step=1):
        for layer in self.layers:
            if optimizer == "adam":
                layer.update_adam(learning_rate, step)
            else:
                layer.update(learning_rate)
        
    def evaluate(self, X, y):
        predicted_probs = self.forward(X)
        loss = categorical_crossentropy(y, predicted_probs)
        acc = accuracy(y, predicted_probs)
        return loss, acc

    def fit(self, X_train, y_train, X_valid, y_valid,
             learning_rate=0.03, batch_size=8, epochs=100, seed=None, verbose=True , patience=None, optimizer="gradient_descent"):
    
        if optimizer not in ["gradient_descent", "adam"]:
            raise ValueError("optimizer must be either 'gradient_descent' or 'adam'")
        
        step = 0

        rng = np.random.default_rng(seed)
        n_samples = X_train.shape[0]

        best_val_loss = float("inf")
        best_weights = None
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            shuffled_indices = rng.permutation(n_samples)
            X_shuffled = X_train[shuffled_indices]
            y_shuffled = y_train[shuffled_indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                predicted_probs = self.forward(X_batch)
                self.backward(y_batch, predicted_probs)
                step += 1
                self.update(learning_rate, optimizer, step=step)

            train_loss, train_acc = self.evaluate(X_train, y_train)
            val_loss, val_acc = self.evaluate(X_valid, y_valid)

            self.history["loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            val_probs = self.forward(X_valid)
            tp, fp, tn, fn = confusion_counts(y_valid, val_probs)
            val_precision = precision(tp, fp)
            val_recall = recall(tp, fn)
            val_f1 = f1_score(val_precision, val_recall)

            self.history["val_precision"].append(val_precision)
            self.history["val_recall"].append(val_recall)
            self.history["val_f1"].append(val_f1)

            if verbose:
                print(f"epoch {epoch}/{epochs} - loss: {train_loss:.4f} - "
                      f"val_loss: {val_loss:.4f} - acc: {train_acc:.4f} - "
                      f"val_acc: {val_acc:.4f} - precision: {val_precision:.4f} - "
                      f"recall: {val_recall:.4f} - f1: {val_f1:.4f}")
            if patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = [(layer.weights.copy(), layer.bias.copy())
                                    for layer in self.layers]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} due to no improvement in validation loss for {patience} epochs.")
                    break

        if best_weights is not None:
            for layer, (weights, bias) in zip(self.layers, best_weights):
                layer.weights = weights
                layer.bias = bias

        return self.history

    def save(self, path):
        layer_sizes = [self.layers[0].weights.shape[0]] + \
            [layer.weights.shape[1] for layer in self.layers]
        activations = [layer.activation_name for layer in self.layers]

        data = {
            "layer_sizes": np.array(layer_sizes),
            "activations": np.array(activations),
        }
        for i, layer in enumerate(self.layers):
            data[f"W{i}"] = layer.weights
            data[f"b{i}"] = layer.bias

        np.savez(path, **data)

    @staticmethod
    def load(path):
        archive = np.load(path)
        layer_sizes = archive["layer_sizes"].tolist()
        activations = archive["activations"].tolist()

        net = Network(layer_sizes, activations)
        for i, layer in enumerate(net.layers):
            layer.weights = archive[f"W{i}"]
            layer.bias = archive[f"b{i}"]
        return net