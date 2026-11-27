"""Upload any file, not only an image, to the bucket used for annotated photos.

`lf_toolkit`'s `upload_image` only accepts PIL images and always picks a
random file name. `upload_file` takes raw bytes, a MIME type and a file name
chosen by the caller, and uses the same storage configuration (see "Annotated
image storage" in the README).
"""
import re

import requests
from lf_toolkit.evaluation.image_upload import (
    get_aws_signed_request,
    get_s3_bucket_uri,
    resolve_upload_backend,
)

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class FileUploadError(Exception):
    """Raised when a file cannot be uploaded."""


def safe_name(name: str) -> str:
    """Turn `name` into a path segment that needs no URL encoding.

    Anything other than letters, digits, `.`, `_` and `-` becomes `_`, and
    leading or trailing dots are removed so `..` cannot climb out of a folder.
    """
    cleaned = _UNSAFE_CHARS.sub("_", str(name)).strip(".")
    return cleaned or "file"


def upload_file(data: bytes, filename: str, mime_type: str, folder: str = "eduvision") -> str:
    """Upload `data` as `<folder>/<filename>` and return its public URL.

    An existing file with the same name is overwritten without warning, so
    callers must make names unique when many evaluations can run at once.
    """
    folder_path = "/".join(safe_name(part) for part in str(folder).split("/") if part)
    filename = safe_name(filename)

    try:
        if resolve_upload_backend() == "gcs":
            # Imported here, not at the top: it is a private toolkit helper,
            # and if it ever disappears only this upload should fail, not
            # every evaluation.
            from lf_toolkit.evaluation.image_upload import _upload_gcs
            return _upload_gcs(folder_path, filename, data, mime_type)

        url = f"{get_s3_bucket_uri().rstrip('/')}/{folder_path}/{filename}"
        signed = get_aws_signed_request(url, data, mime_type).prepare()
        response = requests.request(
            method=signed.method,
            url=signed.url,
            data=signed.body,
            headers=signed.headers,
            timeout=30,
        )
    except Exception as e:
        raise FileUploadError(f"{type(e).__name__}: {e}") from e

    if response.status_code != 200:
        raise FileUploadError(
            f"Upload failed with status code {response.status_code}: {response.text}"
        )

    return url
