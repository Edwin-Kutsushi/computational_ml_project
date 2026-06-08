import torch
import torch.nn as nn
import torch.optim as optim


def create_model(model_class, hidden_dim=None, device="cpu"):
    try:
        if hidden_dim is None:
            model = model_class()
        else:
            model = model_class(hidden_dim=hidden_dim)
    except TypeError:
        model = model_class()

    return model.to(device)


def evaluate_mnist(model, test_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / total

    return avg_loss, accuracy


def transfer_cnn_weights(old_model, new_model):
    with torch.no_grad():

        old_layers = list(old_model.net)
        new_layers = list(new_model.net)

        for old_layer, new_layer in zip(old_layers, new_layers):

            if isinstance(old_layer, nn.Conv2d) and isinstance(new_layer, nn.Conv2d):
                new_layer.weight.copy_(old_layer.weight)
                new_layer.bias.copy_(old_layer.bias)

            elif isinstance(old_layer, nn.Linear) and isinstance(new_layer, nn.Linear):
                rows = min(old_layer.weight.shape[0], new_layer.weight.shape[0])
                cols = min(old_layer.weight.shape[1], new_layer.weight.shape[1])

                new_layer.weight[:rows, :cols] = old_layer.weight[:rows, :cols]
                new_layer.bias[:rows] = old_layer.bias[:rows]

    return new_model


def train_mnist_single_level(
    model_class,
    train_loader,
    test_loader,
    hidden_dim=128,
    epochs=15,
    optimizer_class=optim.SGD,
    optimizer_params=None,
    device="cpu"
):
    if optimizer_params is None:
        optimizer_params = {"lr": 0.05}

    model = create_model(
        model_class=model_class,
        hidden_dim=hidden_dim,
        device=device
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * images.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)

        test_loss, test_acc = evaluate_mnist(
            model,
            test_loader,
            criterion,
            device
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    return {
        "model": model,
        "train_loss": train_losses,
        "test_loss": test_losses,
        "test_acc": test_accuracies,
        "test_accuracy": test_accuracies
    }


def train_mnist_multilevel(
    model_class,
    train_loader,
    test_loader,
    hidden_dims=None,
    epochs_per_level=5,
    optimizer_class=optim.SGD,
    optimizer_params=None,
    device="cpu"
):
    if hidden_dims is None:
        hidden_dims = [32, 64, 128]

    if optimizer_params is None:
        optimizer_params = {"lr": 0.05}

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    test_accuracies = []
    level_history = []

    previous_model = None

    global_epoch = 0
    total_epochs = len(hidden_dims) * epochs_per_level

    for level_idx, hidden_dim in enumerate(hidden_dims):
        print(f"\nStarting level {level_idx + 1}: hidden_dim={hidden_dim}")

        model = create_model(
            model_class=model_class,
            hidden_dim=hidden_dim,
            device=device
        )

        if previous_model is not None:
            model = transfer_cnn_weights(previous_model, model)
            print("Transferred CNN weights from previous level")

        optimizer = optimizer_class(model.parameters(), **optimizer_params)

        for epoch in range(epochs_per_level):
            model.train()
            total_train_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * images.size(0)

            global_epoch += 1

            train_loss = total_train_loss / len(train_loader.dataset)

            test_loss, test_acc = evaluate_mnist(
                model,
                test_loader,
                criterion,
                device
            )

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            level_history.append(hidden_dim)

            print(
                f"Epoch {global_epoch}/{total_epochs} | "
                f"{optimizer_class.__name__} | "
                f"hidden_dim={hidden_dim} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )

        previous_model = model

    return {
        "model": model,
        "train_loss": train_losses,
        "test_loss": test_losses,
        "test_acc": test_accuracies,
        "test_accuracy": test_accuracies,
        "level_history": level_history
    }