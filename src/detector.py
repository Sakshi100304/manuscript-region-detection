from pathlib import Path

import cv2
from ultralytics import YOLO

from .postprocessing import (
    clamp_box,
    prediction_record,
    valid_box
)
from .visualization import draw_predictions


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


class ManuscriptDetector:
    """
    YOLO-based detector for manuscript layout regions.
    """

    def __init__(
        self,
        model_path,
        confidence=0.25,
        image_size=1024
    ):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                "Train the model first and place the trained "
                "model at models/best.pt."
            )

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size

    def predict(self, image):
        """
        Detect manuscript regions in one image.
        """

        height, width = image.shape[:2]

        results = self.model.predict(
            source=image,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False
        )

        predictions = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                coordinates = box.xyxy[0].tolist()

                # Keep coordinates inside image boundaries
                safe_box = clamp_box(
                    coordinates,
                    width,
                    height
                )

                if not valid_box(safe_box):
                    continue

                prediction = prediction_record(
                    class_id,
                    confidence,
                    safe_box
                )

                predictions.append(prediction)

        return predictions

    def predict_and_annotate(self, image):
        """
        Detect regions and create an annotated image.
        """

        predictions = self.predict(image)

        annotated_image = draw_predictions(
            image,
            predictions
        )

        return annotated_image, predictions


def find_images(input_path):
    """
    Find supported images from a single image
    or a folder.
    """

    path = Path(input_path)

    # If input is one image
    if path.is_file():

        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [path]

        return []

    # If input is a folder
    if path.is_dir():

        return sorted(
            file
            for file in path.rglob("*")
            if (
                file.is_file()
                and file.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        )

    return []