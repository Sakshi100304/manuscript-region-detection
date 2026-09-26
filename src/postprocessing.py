CLASS_NAMES = {
    0: "header",
    1: "footer",
    2: "main_text",
    3: "side_text",
    4: "filler"
}


def clamp_box(box, width, height):
    """
    Keep the bounding box inside the image boundaries.
    """

    x1, y1, x2, y2 = box

    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width - 1))
    y2 = max(0, min(int(round(y2)), height - 1))

    return x1, y1, x2, y2


def valid_box(box):
    """
    Check whether the bounding box has a valid area.
    """

    x1, y1, x2, y2 = box

    return x2 > x1 and y2 > y1


def prediction_record(class_id, confidence, box):
    """
    Create one prediction record for JSON output.
    """

    return {
        "label": CLASS_NAMES.get(class_id, f"class_{class_id}"),
        "class_id": class_id,
        "confidence": round(float(confidence), 4),
        "bbox": {
            "x1": box[0],
            "y1": box[1],
            "x2": box[2],
            "y2": box[3]
        }
    }