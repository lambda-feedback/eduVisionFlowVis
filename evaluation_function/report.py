"""Build the stand-alone HTML evaluation report turned on by `html_report`.

The page is one self-contained file: `report_template.html` holds the CSS and
JavaScript, and the evaluation data is embedded as JSON that the script turns
into the report in the browser.
"""
import json
import os
import time
import uuid

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html")
DATA_PLACEHOLDER = "__REPORT_DATA__"

REPORT_FOLDER = "eduvision/reports"
REPORT_MIME_TYPE = "text/html; charset=utf-8"


def report_file_name(now=None):
    """Return the report's file name: `eduvision-report-YYYYMMDD-HHMMSS-<8 hex>.html`.

    The date and time are UTC. The random suffix keeps two reports made in
    the same second from overwriting each other.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now))
    return f"eduvision-report-{stamp}-{uuid.uuid4().hex[:8]}.html"


def build_report_html(data):
    """Return the report page with `data` embedded in it."""
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()

    # The JSON sits inside a <script> element, so `<` must be escaped or a
    # file name such as "</script><script>..." could end the element and run
    # its own code. `>` and `&` are escaped too, for HTML parsers that are
    # stricter than browsers. JSON.parse turns the escapes back into text.
    payload = (
        json.dumps(data, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return template.replace(DATA_PLACEHOLDER, payload, 1)
