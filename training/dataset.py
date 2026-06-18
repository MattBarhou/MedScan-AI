from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
    project_path,
)


def _count_images(folder: Path) -> int:
    return sum(
        1
        for file_path in folder.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_data_folders() -> None:
    splits = {
        "train": project_path(TRAIN_DIR),
        "val": project_path(VAL_DIR),
        "test": project_path(TEST_DIR),
    }

    for split_name, split_path in splits.items():
        if not split_path.exists():
            raise FileNotFoundError(
                f"Missing {split_name} folder: {split_path}\n"
                f"Expected structure: data/{split_name}/Normal/ and data/{split_name}/Disease/"
            )

        for class_name in CLASS_NAMES:
            class_path = split_path / class_name

            if not class_path.exists():
                raise FileNotFoundError(
                    f"Missing class folder: {class_path}\n"
                    f"Each split must contain folders: {', '.join(CLASS_NAMES)}"
                )

            image_count = _count_images(class_path)
            if image_count == 0:
                raise FileNotFoundError(
                    f"No images found in {class_path}\n"
                    f"Add .jpg, .jpeg, or .png files before training."
                )


def get_data_loaders():
    validate_data_folders()

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_dataset = datasets.ImageFolder(project_path(TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(project_path(VAL_DIR), transform=eval_transform)
    test_dataset = datasets.ImageFolder(project_path(TEST_DIR), transform=eval_transform)

    class_names = train_dataset.classes

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, class_names
