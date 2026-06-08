import time
import numpy as np


def _to_list(x):
    if isinstance(x, list):
        return x

    if isinstance(x, tuple):
        return list(x)

    if isinstance(x, np.ndarray):
        return x.tolist()

    if hasattr(x, "detach"):
        return [x.detach().cpu().item()]

    return [x]


def _find_metric(result, metric_type):
    keys = list(result.keys())

    if metric_type == "train_loss":
        preferred = [
            "train_losses",
            "train_loss",
            "train_loss_history"
        ]
        keywords = ["train", "loss"]

    elif metric_type == "test_loss":
        preferred = [
            "test_losses",
            "test_loss",
            "test_loss_history",
            "val_losses",
            "validation_losses"
        ]
        keywords = ["test", "loss"]

    elif metric_type == "test_accuracy":
        preferred = [
            "test_accuracies",
            "test_accuracy",
            "test_acc",
            "accuracy",
            "accuracies",
            "test_acc_history"
        ]
        keywords = ["acc"]

    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

    for key in preferred:
        if key in result:
            return _to_list(result[key])

    for key in keys:
        lower_key = key.lower()
        if all(word in lower_key for word in keywords):
            return _to_list(result[key])

    raise KeyError(
        f"Could not find {metric_type}. Available result keys are: {keys}"
    )


def run_mnist_experiment(
    name,
    train_fn,
    model_class,
    train_loader,
    test_loader,
    optimizer_class,
    optimizer_params=None,
    device="cpu",
    hidden_dim=128,
    epochs=15,
    hidden_dims=None,
    epochs_per_level=5
):
    """
    Runs one MNIST experiment and standardizes the output.

    Single-level experiment:
        uses hidden_dim and epochs

    Multilevel experiment:
        uses hidden_dims and epochs_per_level

    Works with train_mnist_single_level and train_mnist_multilevel
    from multilevel.py.
    """

    if optimizer_params is None:
        optimizer_params = {"lr": 0.05}

    print(f"\nRunning experiment: {name}")

    start_time = time.time()

    if hidden_dims is None:
        result = train_fn(
            model_class=model_class,
            train_loader=train_loader,
            test_loader=test_loader,
            hidden_dim=hidden_dim,
            epochs=epochs,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            device=device
        )

        total_epochs = epochs

    else:
        result = train_fn(
            model_class=model_class,
            train_loader=train_loader,
            test_loader=test_loader,
            hidden_dims=hidden_dims,
            epochs_per_level=epochs_per_level,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            device=device
        )

        total_epochs = len(hidden_dims) * epochs_per_level

    runtime = time.time() - start_time

    train_losses = _find_metric(result, "train_loss")
    test_losses = _find_metric(result, "test_loss")
    test_accuracies = _find_metric(result, "test_accuracy")

    best_idx = int(np.argmax(test_accuracies))

    result["name"] = name
    result["runtime_seconds"] = runtime
    result["epochs"] = total_epochs

    result["train_losses"] = train_losses
    result["test_losses"] = test_losses
    result["test_accuracies"] = test_accuracies

    result["final_train_loss"] = train_losses[-1]
    result["final_test_loss"] = test_losses[-1]
    result["final_test_accuracy"] = test_accuracies[-1]

    result["best_epoch"] = best_idx + 1
    result["best_test_accuracy"] = test_accuracies[best_idx]
    result["best_test_loss"] = test_losses[best_idx]
    result["best_train_loss"] = train_losses[best_idx]

    result["optimizer"] = optimizer_class.__name__
    result["optimizer_params"] = optimizer_params

    print(f"{name} completed in {runtime:.2f} seconds")
    print(f"Final Train Loss: {result['final_train_loss']:.4f}")
    print(f"Final Test Loss: {result['final_test_loss']:.4f}")
    print(f"Final Test Accuracy: {result['final_test_accuracy']:.4f}")
    print(f"Best Epoch: {result['best_epoch']}")
    print(f"Best Test Accuracy: {result['best_test_accuracy']:.4f}")

    return result