import os
from typing import Any
from lf_toolkit.evaluation import Result, Params
from lf_toolkit.evaluation.image_upload import upload_image
from PIL import Image, ImageOps
import io
import requests
import random
import numpy as np
import cv2
import time


# Cache for loaded models by name
_model_cache = {}


def evaluation_function(
    response: Any,
    answer: Any,
    params: Params,
) -> Result:
    start_total = time.time()

    if not response:
        # Note: the first element of a feedback_items tuple is only an
        # internal grouping tag for lf_toolkit.Result - it is never shown
        # to the student, so any visible heading must live in the body text.
        return Result(
            is_correct=False,
            feedback_items=[(
                "\n\n## No Image Provided\n",
                "\n\n### No Image Provided\n\nPlease upload at least one image to be evaluated.\n",
            )],
        )

    draw_images = params.get("draw_images", True)
    model_name = params.get("model_name", "best.pt")
    conf_threshold = params.get("conf_threshold", 0.5)
    # Optional array of class names to restrict detection to. Names are
    # matched against the model's own class names (model.names),
    # case-sensitive but with surrounding whitespace ignored.
    allowed_classes = params.get("allowed_classes", None)

    model_load_start = time.time()

    # Use a dict to cache models by name
    if model_name not in _model_cache:
        # Imported here rather than at module level: importing ultralytics
        # pulls in torch, which takes longer than Lambda's 10 s init window
        # on a cold start. Deferring it lets the IPC server start first.
        from ultralytics import YOLO

        model_dir = os.path.dirname(os.path.abspath(__file__))
        # Strip any directory components so `model_name` can't be used to
        # load a model file from outside the evaluation_function directory.
        safe_model_name = os.path.basename(model_name)
        model_path = os.path.join(model_dir, safe_model_name)
        _model_cache[model_name] = YOLO(model_path)

    model_load_time = time.time() - model_load_start

    model = _model_cache[model_name]

    # Resolve `allowed_classes` names to the class indices YOLO expects.
    # Unmatched names are kept aside so they can be surfaced in debug
    # feedback instead of silently doing nothing.
    class_indices = None
    unmatched_classes = []
    if allowed_classes:
        name_to_idx = {name: idx for idx, name in model.names.items()}
        matched_indices = []
        for name in allowed_classes:
            key = str(name).strip()
            if key in name_to_idx:
                matched_indices.append(name_to_idx[key])
            else:
                unmatched_classes.append(name)
        # An empty list means "match no class" to YOLO, not "don't filter".
        # Only apply the filter if at least one name actually matched.
        if matched_indices:
            class_indices = matched_indices

    feedback_items = []

    feedback_start = time.time()

    def append_feedback(title, text):
        # Use standard Markdown for all feedback items
        # Title as level 2 heading, text as Markdown body
        markdown_title = f"\n\n## {title.strip()}\n"
        markdown_body = "\n\n" + text.strip() + "\n"
        feedback_items.append((markdown_title, markdown_body))

    def get_class_color(class_name):
        random.seed(hash(class_name) % 10000)
        return tuple(random.choices(range(50, 256), k=3))

    def draw_annotations_cv2(img, detections):

        img_cv = np.array(img)

        if img_cv.shape[2] == 4:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)

        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        h, w = img_cv.shape[:2]

        # Every size below is derived from the image's shorter side so the
        # annotations look the same whether the student uploads a 640px
        # thumbnail or a 4032x3024 phone photo. With FONT_HERSHEY_SIMPLEX,
        # font scale 1 is ~22px tall, so short_side / 900 makes the label
        # text roughly 2.5% of the shorter side (~75px on a 3024px photo).
        short_side = min(h, w)
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_font_scale = max(0.4, short_side / 900)
        box_thickness = max(2, round(short_side / 400))

        for det in detections:

            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            color = get_class_color(str(cls))
            color_bgr = (color[2], color[1], color[0])

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color_bgr, box_thickness)

            label = f"{cls}: {conf:.2f}"

            # Shrink the label if it would be wider than the image, so long
            # class names never get cut off at the right edge.
            font_scale = base_font_scale
            text_thickness = max(1, round(font_scale * 2))
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            pad = max(2, round(th * 0.3))
            if tw + 2 * pad > w:
                font_scale *= w / (tw + 2 * pad)
                text_thickness = max(1, round(font_scale * 2))
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
                pad = max(2, round(th * 0.3))

            label_w = tw + 2 * pad
            label_h = th + baseline + 2 * pad

            # Place the label just above the box; if that would go past the
            # top of the image, put it just inside the top of the box instead.
            # Then slide it left if it would run off the right edge.
            label_x = max(0, min(x1, w - label_w))
            label_y = y1 - label_h if y1 - label_h >= 0 else max(0, y1)
            label_y = min(label_y, h - label_h)

            cv2.rectangle(
                img_cv, (label_x, label_y), (label_x + label_w, label_y + label_h),
                color_bgr, -1,
            )

            # Pick black or white text depending on how bright the label
            # background is, so the text always has strong contrast.
            b, g, r = color_bgr
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = (0, 0, 0) if luminance > 150 else (255, 255, 255)

            cv2.putText(
                img_cv,
                label,
                (label_x + pad, label_y + pad + th),
                font,
                font_scale,
                text_color,
                text_thickness,
                cv2.LINE_AA,
            )

        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    def analyze_images(images, draw_images=True):

        per_image = []
        prediction_times = []
        load_times = []
        process_times = []
        draw_times = []

        for image in images:

            load_start = time.time()

            try:

                url = image["url"]

                # Phone photos are often stored sideways with an EXIF
                # orientation tag. Rotate them upright here so the model and
                # the drawn boxes use the same pixels (newer Ultralytics
                # versions apply the tag themselves, older ones don't).
                if url.startswith("file://"):

                    with open(url[7:], "rb") as f:
                        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")

                else:

                    img_data = requests.get(url).content
                    img = ImageOps.exif_transpose(Image.open(io.BytesIO(img_data))).convert("RGB")

            except Exception as e:

                load_times.append(time.time() - load_start)

                per_image.append(
                    {"annotated": None, "detections": [], "load_error": str(e)}
                )
                continue

            load_times.append(time.time() - load_start)

            pred_start = time.time()
            results = model.predict(img, conf=conf_threshold, classes=class_indices)
            prediction_times.append(time.time() - pred_start)

            process_start = time.time()

            detections = []

            for res in results:

                for box in res.boxes:

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = model.names[int(box.cls[0])]

                    detections.append((x1, y1, x2, y2, conf, cls))

            # Most confident detection first, so the feedback list reads top-down.
            detections.sort(key=lambda d: d[4], reverse=True)

            process_times.append(time.time() - process_start)

            annotated = None

            if draw_images:
                draw_start = time.time()
                annotated = draw_annotations_cv2(img.copy(), detections)
                draw_times.append(time.time() - draw_start)
            else:
                draw_times.append(0.0)

            per_image.append(
                {"annotated": annotated, "detections": detections, "load_error": None}
            )

        avg_load_time = np.mean(load_times) if load_times else 0.0
        avg_prediction_time = np.mean(prediction_times) if prediction_times else 0.0
        avg_process_time = np.mean(process_times) if process_times else 0.0
        avg_draw_time = np.mean(draw_times) if draw_times else 0.0

        return (
            per_image,
            avg_load_time,
            avg_prediction_time,
            avg_process_time,
            avg_draw_time,
        )

    analysis_start = time.time()

    (
        per_image,
        avg_load_time,
        avg_prediction_time,
        avg_process_time,
        avg_draw_time,
    ) = analyze_images(response, draw_images)

    analysis_time = time.time() - analysis_start

    upload_times = []

    for idx, info in enumerate(per_image):
        orig_name = response[idx].get("name", f"image_{idx}.jpg")
        detections = info["detections"]
        if info["load_error"]:
            text = f"*Could not load image: {info['load_error']}*"
        elif not detections:
            text = "*No objects detected.*"
        else:
            lines = [
                f"{n}. `{cls}` - confidence `{conf:.2f}`"
                for n, (_, _, _, _, conf, cls) in enumerate(detections, start=1)
            ]
            text = f"**Detected objects:** {len(detections)}\n\n" + "\n".join(lines)
        append_feedback(
            f"Image {idx}",
            f"### Image {orig_name}\n\n"
            f"{text}\n\n"
        )
        img = info["annotated"]
        if draw_images and img is not None:
            upload_start = time.time()
            try:
                url = upload_image(img, "eduvision")
                # add separate feedback for this uploaded annotated image (Markdown image)
                append_feedback(f"Uploaded Image [{idx}]", f"![{orig_name}]({url}) \n\n---\n\n")
            except Exception as e:
                # Only leak the underlying error (e.g. AWS/S3 details) in debug mode.
                detail = f": {e}" if params.get('debug', False) else ""
                append_feedback(f"Uploaded Image [{idx}]", f"*Image upload failed for {orig_name}{detail}* \n\n---\n\n")
            upload_times.append(time.time() - upload_start)
        else:
            append_feedback(f"Separator [{idx}]", " \n\n---\n\n")
            upload_times.append(0.0)

    avg_upload_time = np.mean(upload_times) if upload_times else 0.0

    feedback_time = time.time() - feedback_start

    total_time = time.time() - start_total

    if params.get('debug_response', False):
        try:
            append_feedback("DEBUG Response Structure", f"**Response Structure:** {repr(response)}")
        except Exception as e:
            append_feedback("Failed to print response structure", f"{e}")

    if params.get('debug', False):

        if allowed_classes:
            append_feedback(
                "DEBUG Allowed Classes",
                f"- **Requested:** `{allowed_classes}`\n"
                f"- **Matched indices:** `{class_indices}`\n"
                f"- **Unmatched names:** `{unmatched_classes if unmatched_classes else 'none'}`"
            )

        # include all annotated/uploaded images in debug output
        for idx, info in enumerate(per_image):
            if info["annotated"] is not None:
                name = response[idx].get("name", f"image_{idx}.jpg")
                append_feedback(f'Debug Uploaded Image [{idx}]', f'![{name}]({response[idx]["url"]})')

        append_feedback(
            "DEBUG Times",
            f"| Step | Time (s) |\n|---|---|\n"
            f"| Models load | {model_load_time:.3f} |\n"
            f"| Avg image load | {avg_load_time:.3f} |\n"
            f"| Avg prediction | {avg_prediction_time:.3f} |\n"
            f"| Avg detection process | {avg_process_time:.3f} |\n"
            f"| Avg drawing | {avg_draw_time:.3f} |\n"
            f"| Avg upload | {avg_upload_time:.3f} |\n"
            f"| Analysis | {analysis_time:.3f} |\n"
            f"| Feedback | {feedback_time:.3f} |\n"
            f"| **Total** | **{total_time:.3f}** |\n"
        )

    # There is no grading yet: a response counts as correct as soon as the
    # model detects anything in at least one of the submitted images.
    is_correct = any(info["detections"] for info in per_image)

    return Result(
        is_correct=is_correct,
        feedback_items=feedback_items,
    )
