from typing import Any
from lf_toolkit.preview import Result, Params, Preview

def preview_function(response: Any, params: Params) -> Result:
    """
    Function used to preview a student response.
    ---
    The handler function passes two arguments to preview_function():

    - `response` which is the answer provided by the student.
    - `params` which are any extra parameters that may be useful.

    This evaluation function only accepts image upload responses (a list
    of `{url, name, ...}` objects), which have no symbolic representation
    to preview server-side. The platform already shows the uploaded image
    itself, so this returns an empty preview instead of attempting to
    parse the response as a math expression (which would just surface a
    raw parser error to the student).
    """

    return Result(preview=Preview())
