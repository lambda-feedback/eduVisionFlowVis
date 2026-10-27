import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from .evaluation import Params, evaluation_function


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
        # Unmatched names must be surfaced in debug feedback so typos are easy to catch.
        self.assertIn("not_a_real_class", result["feedback"])
