from pathlib import Path
import argparse
import json
import cv2

from src.detector import ManuscriptDetector, find_images


def run_inference(model_path, input_path, output_dir):
    model_path = Path(model_path)
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # Create output folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check model
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    # Check input
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path not found: {input_path}"
        )

    # Find images
    if input_path.is_file():
        images = [input_path]
    else:
        images = find_images(input_path)

    if not images:
        print("No images found.")
        return

    # Load detector
    detector = ManuscriptDetector(
        model_path,
        confidence=0.25,
        image_size=1024
    )

    print(f"Found {len(images)} image(s).")
    print("Starting inference...\n")

    for image_path in images:

        print(f"Processing: {image_path.name}")

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        # Detect and annotate
        annotated_image, predictions = (
            detector.predict_and_annotate(image)
        )

        # Save annotated image
        output_image_path = (
            output_dir /
            f"{image_path.stem}_result{image_path.suffix}"
        )

        cv2.imwrite(
            str(output_image_path),
            annotated_image
        )

        # Save JSON
        output_json_path = (
            output_dir /
            f"{image_path.stem}.json"
        )

        with open(
            output_json_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                {
                    "image": image_path.name,
                    "predictions": predictions
                },
                file,
                indent=4
            )

        print(
            f"  Image saved: {output_image_path}"
        )

        print(
            f"  JSON saved: {output_json_path}"
        )

        print(
            f"  Regions detected: {len(predictions)}\n"
        )

    print("Inference completed successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Manuscript region detection inference"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to an image or folder of images"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Folder where results will be saved"
    )

    parser.add_argument(
        "--model",
        default="models/best.pt",
        help="Path to trained YOLO model"
    )

    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        input_path=args.input,
        output_dir=args.output
    )