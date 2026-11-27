import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from .evaluation import DEFAULT_MODEL_NAME, Params, _load_model, evaluation_function
from .file_upload import FileUploadError
from .heartbeat import HeartbeatError
from .report_test import embedded_data


class FakeReportStorage:
    """Stands in for upload_file and keeps what was uploaded."""

    def __init__(self):
        self.uploads = []

    def upload(self, data, filename, mime_type, folder="eduvision"):
        self.uploads.append({"data": data, "filename": filename, "mime_type": mime_type, "folder": folder})
        return f"https://example.com/{folder}/{filename}"

    def report_data(self):
        report = next(u for u in self.uploads if u["filename"].endswith(".html"))
        return embedded_data(report["data"].decode("utf-8"))


def local_image_response():
    local_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "evaluation_test.py.jpg"))
    return [{
        "comment": "",
        "name": "evaluation_test.py.jpg",
        "size": os.path.getsize(local_image_path),
        "type": "image/jpeg",
        "url": f"file://{local_image_path}",
    }]


class TestEvaluationFunction(unittest.TestCase):
    """
    TestCase Class used to test the algorithm.
    ---
    Tests are used here to check that the algorithm written
    is working as it should.

    It's best practise to write these tests first to get a
    kind of 'specification' for how your algorithm should
    work, and you should run these tests before committing
    your code to AWS.

    Read the docs on how to use unittest here:
    https://docs.python.org/3/library/unittest.html

    Use evaluation_function() to check your algorithm works
    as it should.
    """
    def test_evaluation_lists_detections(self):
        params = Params(draw_images=False, debug=True)

        result = evaluation_function(local_image_response(), "test_answer", params).to_dict()

        self.assertIsInstance(result, dict)
        self.assertTrue(result["is_correct"])
        self.assertIn("Image evaluation_test.py.jpg", result["feedback"])
        # The test photo shows the flow channel's main section on its own.
        self.assertIn("**Detected objects:** 1", result["feedback"])
        self.assertIn("`Main Section`", result["feedback"])

    def test_default_model_is_obb(self):
        # The default model predicts oriented (rotated) boxes, read from `res.obb`.
        self.assertEqual(_load_model(DEFAULT_MODEL_NAME).task, "obb")

    def test_annotated_image_is_upright(self):
        # The test photo is stored as 4032x3024 with an EXIF tag rotating it
        # to portrait, so the boxes must be drawn on the rotated 3024x4032 image.
        uploaded = []

        def fake_upload(img, folder_name):
            uploaded.append(img)
            return "https://example.com/annotated.jpg"

        with patch("evaluation_function.evaluation.upload_image", fake_upload):
            result = evaluation_function(local_image_response(), "test_answer", Params()).to_dict()

        self.assertEqual(len(uploaded), 1)
        self.assertEqual(uploaded[0].size, (3024, 4032))
        self.assertIn("https://example.com/annotated.jpg", result["feedback"])

    def test_no_detections_reported(self):
        # A plain grey image contains nothing for the model to detect.
        with tempfile.TemporaryDirectory() as tmp_dir:
            blank_path = os.path.join(tmp_dir, "blank.png")
            Image.new("RGB", (640, 640), (128, 128, 128)).save(blank_path)
            response = [{"name": "blank.png", "url": f"file://{blank_path}"}]

            result = evaluation_function(response, "test_answer", Params(draw_images=False)).to_dict()

        self.assertFalse(result["is_correct"])
        self.assertIn("No objects detected", result["feedback"])

    def test_unloadable_image_reported(self):
        response = [{"name": "missing.jpg", "url": "file://does/not/exist.jpg"}]

        result = evaluation_function(response, "test_answer", Params(draw_images=False)).to_dict()

        self.assertFalse(result["is_correct"])
        self.assertIn("Could not load image", result["feedback"])

    def test_non_image_file_reported(self):
        # Pillow can't identify this file, which is the path where Ultralytics
        # falls back to the HEIF plugin before giving up.
        with tempfile.TemporaryDirectory() as tmp_dir:
            text_path = os.path.join(tmp_dir, "notes.jpg")
            with open(text_path, "w") as f:
                f.write("not an image")
            response = [{"name": "notes.jpg", "url": f"file://{text_path}"}]

            result = evaluation_function(response, "test_answer", Params(draw_images=False)).to_dict()

        self.assertFalse(result["is_correct"])
        self.assertIn("Could not load image", result["feedback"])

    def test_no_response_returns_feedback_without_crashing(self):
        result = evaluation_function([], "test_answer", Params()).to_dict()

        self.assertIsInstance(result, dict)
        self.assertFalse(result["is_correct"])
        self.assertIn("feedback", result)
        self.assertIn("No Image Provided", result["feedback"])

    def test_allowed_classes_filters_and_reports_unmatched(self):
        params = Params(
            draw_images=False,
            debug=True,
            allowed_classes=["Pump", "not_a_real_class"],
        )

        result = evaluation_function(local_image_response(), "test_answer", params).to_dict()

        # Only the Main Section is in the photo, so restricting to Pump finds nothing.
        self.assertFalse(result["is_correct"])
        self.assertIn("No objects detected", result["feedback"])
        # Unmatched names must be surfaced in the feedback so typos are easy to catch.
        self.assertIn("**Warning:** Class `not_a_real_class`", result["feedback"])

    def test_allowed_classes_is_case_insensitive(self):
        params = Params(draw_images=False, allowed_classes=["  main section "])

        result = evaluation_function(local_image_response(), "test_answer", params).to_dict()

        self.assertIn("`Main Section`", result["feedback"])
        self.assertNotIn("Warning", result["feedback"])

    def test_allowed_classes_accepts_single_name(self):
        # Restricting to Pump finds nothing, which proves the filter was applied.
        params = Params(draw_images=False, allowed_classes="Pump")

        result = evaluation_function(local_image_response(), "test_answer", params).to_dict()

        self.assertFalse(result["is_correct"])
        self.assertIn("No objects detected", result["feedback"])
        self.assertNotIn("Warning", result["feedback"])

    def test_allowed_classes_all_invalid_detects_everything_with_warning(self):
        params = Params(draw_images=False, allowed_classes=["not_a_real_class"])

        result = evaluation_function(local_image_response(), "test_answer", params).to_dict()

        self.assertIn("`Main Section`", result["feedback"])
        self.assertIn("all classes were detected", result["feedback"])

    def test_invalid_model_name_falls_back_to_default_with_warning(self):
        for model_name in ("missing.pt", "evaluation.py", "", 5):
            with self.subTest(model_name=model_name):
                params = Params(draw_images=False, model_name=model_name)

                result = evaluation_function(local_image_response(), "test_answer", params).to_dict()

                self.assertTrue(result["is_correct"])
                self.assertIn("`Main Section`", result["feedback"])
                self.assertIn("**Warning:** Model", result["feedback"])

    def test_target_detected_is_correct(self):
        params = Params(draw_images=False, target="main section")

        result = evaluation_function(local_image_response(), "test_answer", params).to_dict()

        self.assertTrue(result["is_correct"])
        self.assertNotIn("Warning", result["feedback"])

    def test_target_not_detected_is_incorrect(self):
        # The photo contains the Main Section, but not the required Pump.
        params = Params(draw_images=False, target="Pump")

        result = evaluation_function(local_image_response(), "test_answer", params).to_dict()

        self.assertFalse(result["is_correct"])
        self.assertIn("`Main Section`", result["feedback"])

    def test_invalid_target_is_incorrect_with_warning(self):
        params = Params(draw_images=False, target="not_a_real_class")

        result = evaluation_function(local_image_response(), "test_answer", params).to_dict()

        self.assertFalse(result["is_correct"])
        self.assertIn("**Warning:** Target class `not_a_real_class`", result["feedback"])

    def test_html_report_uploaded_with_fixed_name(self):
        uploads = []

        def fake_upload_file(data, filename, mime_type, folder="eduvision"):
            uploads.append({"data": data, "filename": filename, "mime_type": mime_type, "folder": folder})
            return f"https://example.com/{folder}/{filename}"

        params = Params(draw_images=False, html_report=True, target="Main Section")
        with patch("evaluation_function.evaluation.upload_file", fake_upload_file):
            result = evaluation_function(local_image_response(), "test_answer", params).to_dict()

        self.assertEqual(len(uploads), 1)
        upload = uploads[0]
        self.assertRegex(upload["filename"], r"^eduvision-report-\d{8}-\d{6}-[0-9a-f]{8}\.html$")
        self.assertEqual(upload["folder"], "eduvision/reports")
        self.assertEqual(upload["mime_type"], "text/html; charset=utf-8")
        self.assertIn(f"https://example.com/eduvision/reports/{upload['filename']}", result["feedback"])
        self.assertIn(upload["filename"], result["feedback"])

        data = embedded_data(upload["data"].decode("utf-8"))
        self.assertTrue(data["is_correct"])
        self.assertEqual(data["file_name"], upload["filename"])
        self.assertEqual(data["settings"]["target_class"], "Main Section")
        image = data["images"][0]
        # Sizes are those of the upright photo, the coordinate system of the boxes.
        self.assertEqual((image["width"], image["height"]), (3024, 4032))
        self.assertEqual(image["preview_kind"], "original")
        self.assertEqual([d["class"] for d in image["detections"]], ["Main Section"])
        self.assertEqual(len(image["detections"][0]["corners"]), 4)

    def test_html_report_off_by_default(self):
        fake_upload_file = MagicMock()

        with patch("evaluation_function.evaluation.upload_file", fake_upload_file):
            result = evaluation_function(local_image_response(), "test_answer", Params(draw_images=False)).to_dict()

        fake_upload_file.assert_not_called()
        self.assertNotIn("Evaluation report", result["feedback"])

    def test_html_report_upload_failure_keeps_result(self):
        failing_upload = MagicMock(side_effect=FileUploadError("bucket says no"))
        params = Params(draw_images=False, html_report=True)

        with patch("evaluation_function.evaluation.upload_file", failing_upload):
            result = evaluation_function(local_image_response(), "test_answer", params).to_dict()

        self.assertTrue(result["is_correct"])
        self.assertIn("Evaluation report upload failed", result["feedback"])
        # Storage error details are only shown with `debug`.
        self.assertNotIn("bucket says no", result["feedback"])

    def test_heartbeat_started_and_shown_in_report(self):
        storage = FakeReportStorage()
        started = []

        def fake_start_heartbeat(filename, duration, upload, folder):
            started.append({"filename": filename, "duration": duration, "folder": folder})
            return SimpleNamespace(url=f"https://example.com/{folder}/{filename}", filename=filename,
                                   interval=10, duration=duration)

        params = Params(draw_images=False, html_report=True, heartbeat_seconds=60)
        with patch("evaluation_function.evaluation.upload_file", storage.upload), \
                patch("evaluation_function.evaluation.start_heartbeat", fake_start_heartbeat):
            result = evaluation_function(local_image_response(), "test_answer", params).to_dict()

        self.assertEqual(len(started), 1)
        report_name = storage.uploads[0]["filename"]
        heartbeat_name = report_name.replace(".html", "-heartbeat.json")
        self.assertEqual(started[0], {"filename": heartbeat_name, "duration": 60, "folder": "eduvision/reports"})
        self.assertIn(f"https://example.com/eduvision/reports/{heartbeat_name}", result["feedback"])
        self.assertIn(heartbeat_name, result["feedback"])

        data = storage.report_data()
        self.assertEqual(data["heartbeat"]["file_name"], heartbeat_name)
        self.assertEqual(data["heartbeat"]["interval_seconds"], 10)
        self.assertIsNone(data["heartbeat_problem"])

    def test_heartbeat_that_cannot_start_is_reported(self):
        storage = FakeReportStorage()
        refusing_start = MagicMock(side_effect=HeartbeatError("3 background heartbeats are already running."))

        params = Params(draw_images=False, html_report=True, heartbeat_seconds=60)
        with patch("evaluation_function.evaluation.upload_file", storage.upload), \
                patch("evaluation_function.evaluation.start_heartbeat", refusing_start):
            result = evaluation_function(local_image_response(), "test_answer", params).to_dict()

        self.assertTrue(result["is_correct"])
        self.assertIn("Background heartbeat did not start. 3 background heartbeats", result["feedback"])
        data = storage.report_data()
        self.assertIsNone(data["heartbeat"])
        self.assertIn("already running", data["heartbeat_problem"])

    def test_invalid_heartbeat_seconds_warns(self):
        fake_start = MagicMock()

        with patch("evaluation_function.evaluation.start_heartbeat", fake_start):
            result = evaluation_function(
                local_image_response(), "test_answer", Params(draw_images=False, heartbeat_seconds="soon")
            ).to_dict()

        fake_start.assert_not_called()
        self.assertIn("**Warning:** `heartbeat_seconds` must be a number", result["feedback"])

    def test_target_is_detected_even_if_not_in_allowed_classes(self):
        params = Params(draw_images=False, target="Main Section", allowed_classes=["Pump"])

        result = evaluation_function(local_image_response(), "test_answer", params).to_dict()

        self.assertTrue(result["is_correct"])
        self.assertIn("`Main Section`", result["feedback"])
