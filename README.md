# EduVision Flow Vis

A [Lambda Feedback](https://github.com/lambda-feedback) evaluation function that looks at photos uploaded by a student and lists the objects it recognises in them, such as the parts of a flow visualisation rig.

No computer vision knowledge is needed to use it. If you only want to set it up on a question, read [What this function does](#what-this-function-does), [Parameters](#parameters) and [Available classes](#available-classes).

## What this function does

1. A student uploads one or more photos as their answer.
2. A trained object-detection model ([YOLO](https://docs.ultralytics.com/)) searches each photo for the objects it was trained to recognise.
3. The student gets feedback for each photo:
   - a numbered list of the objects found, most confident first
   - a confidence score for each object, from `0` to `1`, where higher means the model is more sure
   - a copy of the photo with a labelled box drawn around each object

Example of the feedback text for one photo:

```text
Image rig.jpg

Detected objects: 2

1. Main Section - confidence 0.88
2. Pump - confidence 0.71
```

### How the answer is marked

> [!IMPORTANT]
> The answer set on the question is ignored. To require a specific object, use the `target` parameter. Without `target`, **any** detected object makes the response correct.

| Situation | Marked as |
|---|---|
| `target` is set and that class is found in at least one photo | Correct |
| `target` is set and that class is not found in any photo | Incorrect |
| `target` is set to a class that does not exist in the model | Incorrect, with a warning |
| No `target`, and at least one object found in at least one photo | Correct |
| No `target`, and no objects found in any photo | Incorrect |
| No photo submitted | Incorrect, with a message asking for a photo |
| A photo cannot be opened | That photo gets a "Could not load image" message and counts as having no objects |

### Accepted photos

JPEG and PNG photos work, as do HEIC/HEIF photos, the iPhone default. Sideways phone photos are rotated upright automatically.

## Parameters

Parameters are optional settings that change how the function behaves. In Lambda Feedback, set them in the evaluation function parameters of the question's response area, written as JSON. Any parameter you leave out uses its default.

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `target` | text | none | The class that must be found for the response to be correct, for example `"Pump"`. Capital letters do not matter. See [How the answer is marked](#how-the-answer-is-marked). |
| `model_name` | text | `"best.pt"` | Which trained model to use. If the file does not exist, `best.pt` is used and a warning is shown. See [Adding another model](#adding-another-model). |
| `conf_threshold` | number from `0` to `1` | `0.5` | How sure the model must be before it reports an object. Lower it to find more objects, with more mistakes. Raise it to report fewer objects, with fewer mistakes. |
| `allowed_classes` | list of text | none (all classes) | Only look for these classes. See [Filtering classes](#filtering-classes). |
| `draw_images` | `true` or `false` | `true` | Whether to show the student a copy of each photo with boxes drawn on it. Set it to `false` for a text-only list. |
| `debug` | `true` or `false` | `false` | Adds technical details to the feedback: class filter results, upload error details, the original photos and how long each step took. **For testing only: students can see this output.** |
| `debug_response` | `true` or `false` | `false` | Adds the raw submitted data to the feedback. **For testing only: students can see this output.** |

### Examples

Use the default settings:

```json
{}
```

Report only objects the model is at least 70% sure about:

```json
{
  "conf_threshold": 0.7
}
```

Look only for the pump and the lid, without drawing on the photos:

```json
{
  "allowed_classes": ["Pump", "Lid"],
  "draw_images": false
}
```

Require a pump in the photo, with technical details turned on while testing:

```json
{
  "target": "Pump",
  "debug": true
}
```

### Filtering classes

Follow these rules when you use `allowed_classes`:

- **Spelling matters, capital letters do not.** `Pump` and `pump` both work. Spaces before or after a name are ignored.
- **Use names from the model you selected.** Each model has its own classes.

A name that does not exist is ignored, and a warning naming it appears at the top of the feedback. If **no** name in the list exists, the filter is switched off, **all** classes are detected and another warning says so.

`allowed_classes` also affects marking. Without `target`, a photo containing only objects outside your list has no detections, so the response can be marked incorrect. The `target` class is always detected, even if it is not in `allowed_classes`.

### Warnings

Warnings about incorrect parameters appear at the top of the feedback, so **students can see them too**. Test a question's parameters before releasing it to students.

| Problem | What happens |
|---|---|
| `model_name` does not exist | `best.pt` is used instead |
| A class in `allowed_classes` does not exist | That name is ignored |
| No class in `allowed_classes` exists | All classes are detected |
| `target` does not exist in the model | The response is marked incorrect |

## Available classes

The included model recognises the classes below, and only these. Names are listed as the model spells them.

### `best.pt`: flow visualisation rig (default)

Used when `model_name` is not set. It has 20 classes:

| | | |
|---|---|---|
| `Bolts` | `Camber` | `Downstream End Cap` |
| `Downstream Port Covers` | `Fin-Shaped Insert` | `Honeycomb` |
| `Inlet Tube` | `Lid` | `Main Section` |
| `Measuring Jug` | `Needle` | `O Ring` |
| `Outlet Tube` | `Port Covers` | `Pump` |
| `Syringe 1mL` | `Syringe 3mL` | `Towel` |
| `Upstream End Cap` | `Water Container` | |

### Listing classes yourself

If a model file changes, these lists may go out of date. To print the classes a model really has, run this from the project folder after [installing the project](#install-the-project):

```bash
poetry run python -c "from ultralytics import YOLO; print(YOLO('evaluation_function/best.pt').names)"
```

### Adding another model

Put the trained YOLO `.pt` file in the `evaluation_function` folder, then set `model_name` to its file name. Only the file name is used, so models in other folders cannot be loaded. If the file does not exist, `best.pt` is used and a warning is shown.

## Common problems

| What you see | Likely cause | What to do |
|---|---|---|
| "No objects detected" | The object is not in the selected model, `allowed_classes` excludes it, or the model is not sure enough | Check for warnings at the top of the feedback, then try a lower `conf_threshold` such as `0.3` |
| Wrong objects are listed | `conf_threshold` is too low | Raise it, for example to `0.7` |
| A correct-looking photo is marked incorrect | The `target` object was not found, or its confidence is below `conf_threshold` | Check the list of detected objects, then try a lower `conf_threshold` |
| A warning at the top of the feedback | A parameter is incorrect | See [Warnings](#warnings) |
| "Image upload failed" | Storage for annotated photos is not set up | See [Annotated image storage](#annotated-image-storage), or set `"draw_images": false`. The list of objects still works. |

## Running it on your own computer

### Install the project

You need [Python](https://www.python.org) 3.11 or newer and [Poetry](https://python-poetry.org/docs/#installation), which installs everything else. From the project folder, run:

```bash
poetry install
```

### Run the tests

```bash
poetry run pytest
```

### Try it on your own photo

Save this as `try_photo.py` in the project folder and change the photo path and parameters:

```python
from pathlib import Path

from lf_toolkit.evaluation import Params

from evaluation_function.evaluation import evaluation_function

photo = Path("evaluation_function/evaluation_test.py.jpg").resolve()
response = [{"name": photo.name, "url": f"file://{photo}"}]
params = Params(draw_images=False, conf_threshold=0.5)

result = evaluation_function(response, None, params).to_dict()

print("Correct:", result["is_correct"])
print(result["feedback"])
```

Then run it:

```bash
poetry run python try_photo.py
```

> [!NOTE]
> `evaluation_function.dev` does not work with this function. It passes plain text as the response, and this function needs a list of images. Use the script above instead.

### Annotated image storage

Annotated photos are uploaded to cloud storage so that students can see them, which requires these environment variables. A local `.env` file also works, but never commit it. Without them, the upload fails and the student sees a message instead of the annotated photo.

| Storage | Environment variables |
|---|---|
| Amazon S3 | `S3_BUCKET_URI`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally `AWS_SESSION_TOKEN` and `AWS_REGION` (default `eu-west-2`) |
| Google Cloud Storage | `GCS_BUCKET`, and optionally `GCS_PUBLIC_BASE_URL`. Requires the toolkit's optional `gcs` extra. |

`IMAGE_UPLOAD_BACKEND` can be set to `s3` or `gcs` to choose one explicitly. Otherwise, setting `GCS_BUCKET` selects Google Cloud Storage and anything else uses S3.

For local testing, `draw_images=False` avoids the upload.

## Development

### Repository structure

```bash
evaluation_function/main.py             # entry point that starts the function
evaluation_function/evaluation.py       # detection and feedback logic
evaluation_function/evaluation_test.py  # evaluation tests
evaluation_function/preview.py          # preview (empty, because the response is an image)
evaluation_function/preview_test.py     # preview tests
evaluation_function/best.pt             # default model

config.json                             # deployment configuration
```

### Build and run the Docker image

```bash
docker build -t eduvision-flow-vis .
docker run -it --rm -p 8080:8080 eduvision-flow-vis
```

This starts the function and exposes it on port `8080`, using [Shimmy](https://github.com/lambda-feedback/shimmy) to handle HTTP requests.

To check that the function starts on its own, without Shimmy:

```bash
docker run -it --rm eduvision-flow-vis python -m evaluation_function.main
```

## Deployment

The function name is set by `EvaluationFunctionName` in [`config.json`](config.json) and must be unique within Lambda Feedback.

- **Staging:** every push to `main` runs the [staging workflow](.github/workflows/staging-deploy.yml).
- **Production:** [create a release request](https://github.com/lambda-feedback/eduVisionFlowVis/issues/new?template=release-request.yml). The release is made with the [production workflow](.github/workflows/production-deploy.yml).

If the function works locally but not in Docker, check that new Python packages are in `pyproject.toml`, new system packages are in the `Dockerfile`, and model files are in `evaluation_function`.
