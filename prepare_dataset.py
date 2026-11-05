
import shutil
from pathlib import Path
from collections import Counter

# =====================================================
# PREPARE DATASET FOR SMART WEATHER CLASSIFICATION
# Classes: Cloudy and Sunny
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "dataset_cls"

SPLITS = ["train", "val"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def load_classes():

    classes_file = SOURCE_DIR / "train" / "classes.txt"

    if not classes_file.exists():
        raise FileNotFoundError(
            f"Classes file not found: {classes_file}"
        )

    classes = [
        line.strip().lower()
        for line in classes_file.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    if classes != ["cloudy", "sunny"]:
        raise ValueError(
            "Expected class order:\n"
            "0 = cloudy\n"
            "1 = sunny\n"
            f"Found: {classes}"
        )

    return classes


def prepare_split(split, classes):

    source_folder = SOURCE_DIR / split

    if not source_folder.is_dir():
        raise FileNotFoundError(
            f"Missing folder: {source_folder}"
        )

    counts = Counter()
    skipped = 0

    for image_path in source_folder.iterdir():

        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in IMAGE_EXTENSIONS
        ):
            continue

        label_path = image_path.with_suffix(".txt")

        if not label_path.exists():
            print(f"Skipping missing label: {image_path.name}")
            skipped += 1
            continue

        try:

            label_lines = [
                line.strip()
                for line in label_path.read_text(
                    encoding="utf-8-sig"
                ).splitlines()
                if line.strip()
            ]

            if not label_lines:
                print(f"Skipping empty label: {image_path.name}")
                skipped += 1
                continue

            class_ids = set()

            for line in label_lines:

                parts = line.split()

                class_id = int(parts[0])

                if class_id < 0 or class_id >= len(classes):
                    raise ValueError(
                        f"Invalid class ID {class_id}"
                    )

                class_ids.add(class_id)

            # Avoid assigning an image containing multiple
            # weather classes to an incorrect category.

            if len(class_ids) != 1:

                print(
                    f"Skipping mixed-class image: "
                    f"{image_path.name}"
                )

                skipped += 1
                continue

            class_id = next(iter(class_ids))

            class_name = classes[class_id].capitalize()

            destination_folder = (
                OUTPUT_DIR / split / class_name
            )

            destination_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            destination_path = (
                destination_folder / image_path.name
            )

            shutil.copy2(
                image_path,
                destination_path
            )

            counts[class_name] += 1

        except (ValueError, IndexError) as error:

            print(
                f"Invalid annotation for {image_path.name}: {error}"
            )

            skipped += 1

    print(f"\n{split.upper()} DATASET RESULTS")

    for class_name in ["Cloudy", "Sunny"]:

        print(
            f"{class_name}: {counts[class_name]} images"
        )

    print(f"Skipped images: {skipped}")

    return counts


def main():

    print("=" * 55)
    print("SMART WEATHER CLASSIFICATION DATASET PREPARATION")
    print("=" * 55)

    classes = load_classes()

    print("\nClass mapping:")

    for index, class_name in enumerate(classes):

        print(f"{index} = {class_name.capitalize()}")

    total_counts = {}

    for split in SPLITS:

        total_counts[split] = prepare_split(
            split,
            classes
        )

    print("\n" + "=" * 55)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 55)

    for split in SPLITS:

        print(f"\n{split.upper()}:")

        for class_name in ["Cloudy", "Sunny"]:

            print(
                f"{class_name}: "
                f"{total_counts[split][class_name]}"
            )

    print(f"\nOutput folder: {OUTPUT_DIR}")

    print("\nNew classification dataset is ready!")


if __name__ == "__main__":

    main()
