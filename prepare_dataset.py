from pathlib import Path
from PIL import Image
import numpy as np
import shutil
import random

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SOURCE = BASE_DIR / "dataset" / "extracted" / "Dataset"

OUTPUT_IMAGES = BASE_DIR / "data" / "images"
OUTPUT_LABELS = BASE_DIR / "data" / "labels"

TRAIN_SRC = SOURCE / "train"
TEST_SRC = SOURCE / "test"

# Assignment classes
CLASS_NAMES = {
    0: "header",
    1: "footer",
    2: "main_text",
    3: "side_text",
    4: "filler",
}

# GT RGB colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)       # Decoration
GREEN = (0, 255, 0)     # Filler
BLUE = (0, 0, 255)      # Text
YELLOW = (255, 255, 0)  # Main Body
CYAN = (0, 255, 255)    # Drop Caps


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():
    for split in ["train", "val", "test"]:
        (OUTPUT_IMAGES / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_LABELS / split).mkdir(parents=True, exist_ok=True)


# ============================================================
# RGB MASK
# ============================================================

def color_mask(image, color):
    return np.all(image == color, axis=2)


# ============================================================
# SIMPLE CONNECTED COMPONENTS
# ============================================================

def connected_components(mask):
    """
    Find connected components using only NumPy.

    Returns:
        List of bounding boxes:
        (x1, y1, x2, y2, area)
    """

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    components = []

    for y in range(height):
        for x in range(width):

            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True

            min_x = max_x = x
            min_y = max_y = y
            area = 0

            while stack:
                cy, cx = stack.pop()

                area += 1

                min_x = min(min_x, cx)
                max_x = max(max_x, cx)

                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                neighbors = [
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ]

                for ny, nx in neighbors:

                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if area >= 100:
                components.append(
                    (min_x, min_y, max_x, max_y, area)
                )

    return components


# ============================================================
# BLUE TEXT CLASSIFICATION
# ============================================================

def classify_blue_component(box, image_width, image_height):
    """
    Classify blue Text regions into the assignment classes
    using their position on the manuscript page.
    """

    x1, y1, x2, y2, area = box

    # Normalized bounding-box coordinates
    left = x1 / image_width
    right = x2 / image_width
    top = y1 / image_height

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------
    # Region begins near the top of the page.
    if top < 0.15:
        return 0

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------
    # Region begins in the lower 30% of the page.
    if top > 0.70:
        return 1

    # --------------------------------------------------------
    # Side text
    # --------------------------------------------------------
    # Region lies near the left or right page margin.
    if right < 0.35 or left > 0.65:
        return 3

    # --------------------------------------------------------
    # Main text
    # --------------------------------------------------------
    return 2

# ============================================================
# CREATE YOLO LABELS
# ============================================================

def create_labels(image_path, gt_path, label_path):

    image = Image.open(image_path).convert("RGB")
    gt = Image.open(gt_path).convert("RGB")

    image_array = np.array(image)
    gt_array = np.array(gt)

    height, width = gt_array.shape[:2]

    boxes = []

    # --------------------------------------------------------
    # Main Body -> main_text
    # --------------------------------------------------------

    main_mask = color_mask(gt_array, YELLOW)

    for x1, y1, x2, y2, area in connected_components(main_mask):

        boxes.append(
            (2, x1, y1, x2, y2)
        )

    # --------------------------------------------------------
    # Filler + Decoration + Drop Caps -> filler
    # --------------------------------------------------------

    filler_mask = (
        color_mask(gt_array, GREEN)
        | color_mask(gt_array, RED)
        | color_mask(gt_array, CYAN)
    )

    for x1, y1, x2, y2, area in connected_components(filler_mask):

        boxes.append(
            (4, x1, y1, x2, y2)
        )

    # --------------------------------------------------------
    # Blue Text -> header/footer/side_text/main_text
    # --------------------------------------------------------

    blue_mask = color_mask(gt_array, BLUE)

    blue_components = connected_components(blue_mask)

    for component in blue_components:

        x1, y1, x2, y2, area = component

        class_id = classify_blue_component(
            component,
            width,
            height
        )

        boxes.append(
            (class_id, x1, y1, x2, y2)
        )

    # --------------------------------------------------------
    # Write YOLO format
    # --------------------------------------------------------

    with open(label_path, "w", encoding="utf-8") as f:

        for class_id, x1, y1, x2, y2 in boxes:

            # Clip boxes to image boundaries
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # YOLO format
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height

            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height

            f.write(
                f"{class_id} "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}\n"
            )


# ============================================================
# PROCESS DATASET
# ============================================================

def process_split(source_dir, split, image_files):

    print(f"\nProcessing {split}: {len(image_files)} images")

    for index, image_path in enumerate(image_files, start=1):

        gt_path = source_dir / "GT" / (
            image_path.stem + ".gif"
        )

        if not gt_path.exists():
            print(f"Skipping missing GT: {image_path.name}")
            continue

        output_image = OUTPUT_IMAGES / split / image_path.name
        output_label = OUTPUT_LABELS / split / (
            image_path.stem + ".txt"
        )

        shutil.copy2(image_path, output_image)

        create_labels(
            image_path,
            gt_path,
            output_label
        )

        if index % 50 == 0:
            print(f"Processed {index}/{len(image_files)}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("==============================================")
    print(" Manuscript Dataset Preparation")
    print("==============================================")

    create_directories()

    train_images = sorted(
        TRAIN_SRC.glob("images/*.jpg")
    )

    test_images = sorted(
        TEST_SRC.glob("images/*.jpg")
    )

    # Only keep images that have matching GT
    train_images = [
        p for p in train_images
        if (TRAIN_SRC / "GT" / f"{p.stem}.gif").exists()
    ]

    test_images = [
        p for p in test_images
        if (TEST_SRC / "GT" / f"{p.stem}.gif").exists()
    ]

    print(f"\nMatched training images: {len(train_images)}")
    print(f"Matched test images: {len(test_images)}")

    # --------------------------------------------------------
    # Split training data into train / validation
    # --------------------------------------------------------

    random.seed(42)
    random.shuffle(train_images)

    split_index = int(len(train_images) * 0.8)

    train_split = train_images[:split_index]
    val_split = train_images[split_index:]

    print(f"Training split: {len(train_split)}")
    print(f"Validation split: {len(val_split)}")
    print(f"Test split: {len(test_images)}")

    process_split(
        TRAIN_SRC,
        "train",
        train_split
    )

    process_split(
        TRAIN_SRC,
        "val",
        val_split
    )

    process_split(
        TEST_SRC,
        "test",
        test_images
    )

    print("\n==============================================")
    print("Dataset preparation completed!")
    print("==============================================")


if __name__ == "__main__":
    main()