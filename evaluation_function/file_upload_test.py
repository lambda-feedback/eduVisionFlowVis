import os
import unittest
from unittest.mock import MagicMock, patch

from .file_upload import FileUploadError, safe_name, upload_file

S3_ENV = {
    "IMAGE_UPLOAD_BACKEND": "s3",
    "S3_BUCKET_URI": "https://bucket.example.com/",
    "AWS_ACCESS_KEY_ID": "test-key",
    "AWS_SECRET_ACCESS_KEY": "test-secret",
}


class TestSafeName(unittest.TestCase):
    def test_keeps_safe_names(self):
        self.assertEqual(safe_name("eduvision-report-20261127-120000-a1b2c3d4.html"),
                         "eduvision-report-20261127-120000-a1b2c3d4.html")

    def test_replaces_unsafe_characters(self):
        self.assertEqual(safe_name("zdjęcie stołu?#.jpg"), "zdj_cie_sto_u_.jpg")

    def test_cannot_climb_out_of_folder(self):
        self.assertEqual(safe_name("../../secret.html"), "_.._secret.html")
        self.assertEqual(safe_name(".."), "file")


class TestUploadFile(unittest.TestCase):
    def test_uploads_to_s3_with_chosen_name_and_type(self):
        ok = MagicMock(status_code=200)
        with patch.dict(os.environ, S3_ENV), \
                patch("evaluation_function.file_upload.requests.request", return_value=ok) as request:
            url = upload_file(b"<p>hi</p>", "my report.html", "text/html; charset=utf-8", "eduvision/reports")

        self.assertEqual(url, "https://bucket.example.com/eduvision/reports/my_report.html")
        kwargs = request.call_args.kwargs
        self.assertEqual(kwargs["method"], "PUT")
        self.assertEqual(kwargs["url"], url)
        self.assertEqual(kwargs["data"], b"<p>hi</p>")
        self.assertEqual(kwargs["headers"]["Content-Type"], "text/html; charset=utf-8")
        self.assertIn("Authorization", kwargs["headers"])

    def test_rejected_upload_raises(self):
        forbidden = MagicMock(status_code=403, text="AccessDenied")
        with patch.dict(os.environ, S3_ENV), \
                patch("evaluation_function.file_upload.requests.request", return_value=forbidden):
            with self.assertRaisesRegex(FileUploadError, "403"):
                upload_file(b"x", "a.html", "text/html")

    def test_missing_configuration_raises(self):
        with patch.dict(os.environ, {"IMAGE_UPLOAD_BACKEND": "s3"}, clear=True):
            with self.assertRaisesRegex(FileUploadError, "S3_BUCKET_URI"):
                upload_file(b"x", "a.html", "text/html")
