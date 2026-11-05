import os
from pathlib import Path
from ultralytics import YOLO


# =====================================================
# SMART WEATHER CLASSIFICATION USING YOLOv8
# Classes: Cloudy and Sunny
# =====================================================

# Dataset directory
DATASET_PATH = Path("dataset_cls")

# Training configuration
MODEL_NAME = "yolov8n-cls.pt"
EPOCHS = 50
IMAGE_SIZE = 224
BATCH_SIZE = 8
PATIENCE = 10

PROJECT_NAME = "runs"
EXPERIMENT_NAME = "smart_weather"

EXPECTED_CLASSES = ["Cloudy", "Sunny"]


def check_dataset():
    """Check that training and validation images exist."""

    print("\nChecking dataset...")

    required_folders = [
        DATASET_PATH / "train" / "Cloudy",
        DATASET_PATH / "train" / "Sunny",
        DATASET_PATH / "val" / "Cloudy",
        DATASET_PATH / "val" / "Sunny",
    ]

    for folder in required_folders:
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing dataset folder: {folder}\n"
                "Please arrange your images in the required folders."
            )

    valid_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    counts = {}

    for split in ["train", "val"]:
        counts[split] = {}

        for class_name in EXPECTED_CLASSES:
            folder = DATASET_PATH / split / class_name

            images = [
                p for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in valid_extensions
            ]

            counts[split][class_name] = len(images)

            if not images:
                raise ValueError(
                    f"No images found in {folder}"
                )

            print(
                f"{split}/{class_name}: {len(images)} images"
            )

    total_images = sum(
        counts[split][class_name]
        for split in counts
        for class_name in EXPECTED_CLASSES
    )

    print(f"\nTotal images found: {total_images}")

    return counts


def train_weather_model():
    """Train the YOLO image-classification model."""

    try:
        counts = check_dataset()

        # Warn if the training split is very small
        training_total = sum(counts["train"].values())

        if training_total < 20:
            raise ValueError(
                "Too few training images. "
                "Please add more images before training."
            )

        print("\nLoading pretrained YOLO classification model...")

        model = YOLO(MODEL_NAME)

        print("\nStarting weather classification training...")
        print(f"Epochs: {EPOCHS}")
        print(f"Image size: {IMAGE_SIZE}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Patience: {PATIENCE}")

        # Train model
        model.train(
            data=str(DATASET_PATH.resolve()),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            workers=0,
            patience=PATIENCE,
            project=PROJECT_NAME,
            name=EXPERIMENT_NAME,
            exist_ok=True,
            seed=42,
            plots=True,
            verbose=True
        )

        best_model_path = (
            Path(PROJECT_NAME)
            / EXPERIMENT_NAME
            / "weights"
            / "best.pt"
        )

        print("\n" + "=" * 50)
        print("TRAINING COMPLETED")
        print("=" * 50)

        if best_model_path.exists():
            print(f"Best model: {best_model_path}")

            print("\nEvaluating the best model...")

            best_model = YOLO(str(best_model_path))

            metrics = best_model.val(
                data=str(DATASET_PATH.resolve()),
                split="val",
                imgsz=IMAGE_SIZE,
                workers=0
            )

            print("\nValidation results:")
            print(
                "Top-1 accuracy:",
                f"{metrics.top1:.4f}"
            )
            print(
                "Top-5 accuracy:",
                f"{metrics.top5:.4f}"
            )

            print("\nUse this model in main.py:")
            print(str(best_model_path))

        else:
            print(
                "Training finished, but best.pt was not found. "
                "Check the training output for errors."
            )

    except (FileNotFoundError, ValueError) as error:
        print(f"\nDATASET ERROR:\n{error}")

    except Exception as error:
        print(f"\nTRAINING ERROR:\n{error}")
        raise


if __name__ == "__main__":
    train_weather_model()
