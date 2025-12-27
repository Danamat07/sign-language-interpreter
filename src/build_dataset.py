"""
Builds a clean dataset for American Sign Language alphabet recognition.

The script:
- reads images from a raw ASL dataset
- excludes dynamic letters (J and Z)
- shuffles images per class
- splits data into train / validation / test sets
- verifies minimum number of images per class

Final split:
- 70% training
- 15% validation
- 15% testing
"""

import random
import shutil
from pathlib import Path

# ==============================
# PATHS
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "asl_dataset_raw"
OUTPUT_DIR = BASE_DIR / "data" / "asl_dataset"

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
TEST_DIR = OUTPUT_DIR / "test"

# ==============================
# PARAMETERS
# ==============================
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

MIN_IMAGES_PER_CLASS = 200

EXCLUDED_CLASSES = {"J", "Z"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ==============================
# HELPERS
# ==============================
def is_image(file: Path) -> bool:
    return file.suffix.lower() in IMAGE_EXTENSIONS


# ==============================
# DATASET BUILDER
# ==============================
def build_dataset() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    TRAIN_DIR.mkdir(parents=True)
    VAL_DIR.mkdir(parents=True)
    TEST_DIR.mkdir(parents=True)

    classes = sorted(
        directory.name
        for directory in RAW_DIR.iterdir()
        if directory.is_dir() and directory.name not in EXCLUDED_CLASSES
    )

    print(f"Classes included ({len(classes)}): {classes}\n")

    for cls in classes:
        class_path = RAW_DIR / cls
        images = [img for img in class_path.iterdir() if is_image(img)]

        if len(images) < MIN_IMAGES_PER_CLASS:
            raise ValueError(
                f"Class '{cls}' contains only {len(images)} images. "
                f"Minimum required is {MIN_IMAGES_PER_CLASS}."
            )

        random.shuffle(images)

        total = len(images)
        train_end = int(total * TRAIN_SPLIT)
        val_end = train_end + int(total * VAL_SPLIT)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        (TRAIN_DIR / cls).mkdir(parents=True)
        (VAL_DIR / cls).mkdir(parents=True)
        (TEST_DIR / cls).mkdir(parents=True)

        for img in train_images:
            shutil.copy(img, TRAIN_DIR / cls / img.name)

        for img in val_images:
            shutil.copy(img, VAL_DIR / cls / img.name)

        for img in test_images:
            shutil.copy(img, TEST_DIR / cls / img.name)

        print(
            f"{cls}: "
            f"{len(train_images)} train | "
            f"{len(val_images)} val | "
            f"{len(test_images)} test"
        )

    print("\nDataset successfully created.")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Validation directory: {VAL_DIR}")
    print(f"Test directory: {TEST_DIR}")


if __name__ == "__main__":
    build_dataset()
