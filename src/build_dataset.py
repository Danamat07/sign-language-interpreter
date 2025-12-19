"""
Builds a clean dataset for American Sign Language alphabet recognition.
The script reads images from a raw dataset, excludes dynamic letters (J and Z),
shuffles the data, splits it into training and test sets (80/20),
and verifies that each class contains a sufficient number of images.
"""

import random
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "asl_dataset_raw"
OUTPUT_DIR = BASE_DIR / "data" / "asl_dataset"

TRAIN_DIR = OUTPUT_DIR / "train"
TEST_DIR = OUTPUT_DIR / "test"

TRAIN_SPLIT = 0.8
MIN_IMAGES_PER_CLASS = 200

EXCLUDED_CLASSES = {"J", "Z"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def is_image(file: Path) -> bool:
    return file.suffix.lower() in IMAGE_EXTENSIONS


def build_dataset() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    TRAIN_DIR.mkdir(parents=True)
    TEST_DIR.mkdir(parents=True)

    classes = sorted(
        directory.name
        for directory in RAW_DIR.iterdir()
        if directory.is_dir() and directory.name not in EXCLUDED_CLASSES
    )

    print(f"Classes included ({len(classes)}): {classes}")

    for cls in classes:
        class_path = RAW_DIR / cls
        images = [img for img in class_path.iterdir() if is_image(img)]

        if len(images) < MIN_IMAGES_PER_CLASS:
            raise ValueError(
                f"Class '{cls}' contains only {len(images)} images. "
                f"Minimum required is {MIN_IMAGES_PER_CLASS}."
            )

        random.shuffle(images)

        split_index = int(len(images) * TRAIN_SPLIT)
        train_images = images[:split_index]
        test_images = images[split_index:]

        (TRAIN_DIR / cls).mkdir(parents=True)
        (TEST_DIR / cls).mkdir(parents=True)

        for img in train_images:
            shutil.copy(img, TRAIN_DIR / cls / img.name)

        for img in test_images:
            shutil.copy(img, TEST_DIR / cls / img.name)

        print(
            f"{cls}: {len(train_images)} training images, "
            f"{len(test_images)} test images"
        )

    print("Dataset successfully created.")
    print(f"Training data directory: {TRAIN_DIR}")
    print(f"Test data directory: {TEST_DIR}")


if __name__ == "__main__":
    build_dataset()
