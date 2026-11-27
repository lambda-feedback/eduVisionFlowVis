import json
import re
import unittest

from .report import TEMPLATE_PATH, build_report_html, report_file_name

DATA_SCRIPT = re.compile(r'<script id="report-data" type="application/json">(.*?)</script>', re.S)


def embedded_data(html):
    return json.loads(DATA_SCRIPT.search(html).group(1))


class TestReport(unittest.TestCase):
    def test_file_name_format(self):
        name = report_file_name(0)
        self.assertRegex(name, r"^eduvision-report-19700101-000000-[0-9a-f]{8}\.html$")
        self.assertNotEqual(name, report_file_name(0))

    def test_data_is_embedded(self):
        data = {"is_correct": True, "images": [{"name": "rig.jpg", "detections": []}]}

        html = build_report_html(data)

        self.assertEqual(embedded_data(html), data)
        self.assertNotIn("__REPORT_DATA__", html)

    def test_file_names_cannot_end_the_data_script(self):
        evil = "</script><script>alert(1)</script>&<!--"

        html = build_report_html({"images": [{"name": evil}]})

        with open(TEMPLATE_PATH, encoding="utf-8") as f:
            template = f.read()
        self.assertEqual(html.count("</script>"), template.count("</script>"))
        self.assertEqual(embedded_data(html)["images"][0]["name"], evil)
