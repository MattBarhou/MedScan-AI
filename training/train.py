import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models
from torchvision.models import ResNet50_Weights

from config import LEARNING_RATE, MODEL_SAVE_PATH, NUM_EPOCHS, project_path
from dataset import get_data_loaders


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(train_loader.dataset)


def get_class_weights(train_loader, device):
    """Give Normal class more weight because it has fewer training images."""
    targets = torch.tensor(train_loader.dataset.targets)
    class_counts = torch.bincount(targets).float()

    # Inverse frequency weighting helps reduce bias toward the majority class.
    weights = len(targets) / (len(class_counts) * class_counts)
    return weights.to(device)


def evaluate_model(model, data_loader, device, class_names):
    model.eval()
    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            for label, prediction in zip(labels, predictions):
                class_name = class_names[label.item()]
                class_total[class_name] += 1
                if label == prediction:
                    class_correct[class_name] += 1

    overall_correct = sum(class_correct.values())
    overall_total = sum(class_total.values())
    overall_accuracy = overall_correct / overall_total if overall_total else 0.0

    per_class = {
        name: class_correct[name] / class_total[name] if class_total[name] else 0.0
        for name in class_names
    }

    return overall_accuracy, per_class


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _, class_names = get_data_loaders()
    print(f"Classes: {class_names}")

    model = build_model(len(class_names)).to(device)
    class_weights = get_class_weights(train_loader, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.fc.parameters(), lr=LEARNING_RATE)

    print(f"Class weights: {dict(zip(class_names, class_weights.tolist()))}")

    save_path = project_path(MODEL_SAVE_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_accuracy = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_accuracy, val_per_class = evaluate_model(
            model, val_loader, device, class_names
        )

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"- Training Loss: {train_loss:.4f} "
            f"- Validation Accuracy: {val_accuracy:.4f}"
        )
        for class_name, class_accuracy in val_per_class.items():
            print(f"  {class_name} accuracy: {class_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                },
                save_path,
            )
            print(f"  Saved best model to {save_path}")
            print(f"  Class order: {class_names}")

    print(f"\nTraining complete. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
