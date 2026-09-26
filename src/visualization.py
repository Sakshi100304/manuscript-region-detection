import cv2


def draw_predictions(image, predictions):
    """
    Draw bounding boxes, labels, and confidence scores
    on the manuscript image.
    """

    output = image.copy()

    for prediction in predictions:
        bbox = prediction["bbox"]

        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        label = prediction["label"]
        confidence = prediction["confidence"]

        # Draw bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        # Label text
        text = f"{label} {confidence:.2f}"

        text_y = max(20, y1 - 8)

        cv2.putText(
            output,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return output