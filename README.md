# MatrixVision

https://github.com/sibistean95/dmc_detection_python.git

A Python library for **Data Matrix Code (DMC) detection and decoding** in images, built on OpenCV and NumPy.

The pipeline locates Data Matrix codes in a frame (candidate extraction → L-finder pattern detection → validation → dashed-border detection → precise border fitting), rectifies each one, then estimates the module grid, samples it, and decodes the ECC200 payload.

## Features

* Detect one or more Data Matrix codes in a BGR image and recover their precise quadrilateral location.
* Decode located codes to their text payload.
* Fully configurable pipeline via typed config objects (no magic kwargs).
* Optional, pluggable debug visualization — silent by default, opt-in OpenCV windows for inspection.
* A single public entry point: `import dmc`.

## Installation

Requires Python 3.10+.

```bash
pip install -e .
```

On headless Linux (e.g. CI), `opencv-python` needs system GL libraries:

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

## Quick start

```python
import cv2 as cv
import matrixvision

frame = cv.imread("test\_images/dmc\_sample.jpg")   # BGR image (np.ndarray)

for decoded in matrixvision.detect\_and\_decode(frame):
    print(decoded.text)                            # the payload
    print(decoded.detection.precise\_location.ordered\_vertices())  # corner points
```

`detect\_and\_decode` returns a list of `Decoded` objects (empty if nothing was found). The input must be a BGR `numpy.ndarray`; passing `None`, a non-array (e.g. a PIL image), or a single-channel image raises `ValueError`.

## Per-image configuration

Detection parameters that work for one image often differ for another (lighting, noise, module size, surface). Rather than a single global default, pass a `DetectorConfig` tuned for the input:

```python
from matrixvision import DetectorConfig, BorderFitterConfig

config = DetectorConfig(
    smoothing=11,
    noisy\_surface=True,
    canny\_percentile=90.0,
    border\_fitter\_config=BorderFitterConfig(
        dilate\_size=30,
        blob\_min\_area=50,
        win\_in=50,
        win\_out=30,
        ransac\_max\_pts\_outside=40,
        ransac\_inlier\_threshold=0.9,
    ),
)

results = matrixvision.detect\_and\_decode(frame, detector\_config=config)
```

A practical workflow is to keep a small table of `(image kind, config)` pairs — see `tests/cases.py` for a worked example mapping each test image to the config that decodes it.

## Debugging

By default the library is silent (no windows, no stdout). To inspect each stage visually, pass an OpenCV debug sink:

```python
from matrixvision import CvDebugSink

results = matrixvision.detect\_and\_decode(frame, debug=CvDebugSink())  # opens OpenCV windows
```

## Public API

|Function|Returns|Purpose|
|-|-|-|
|`detect(image, config=DetectorConfig(), debug=NullSink())`|`list\[DetectionResult]`|Locate codes only|
|`decode(image, detection, config=DecoderConfig(), debug=NullSink())`|`Decoded \| None`|Decode one located code|
|`detect\_and\_decode(image, detector\_config=DetectorConfig(), decoder\_config=DecoderConfig(), debug=NullSink())`|`list\[Decoded]`|Locate and decode in one call|

### Result types

* **`DetectionResult`** — `candidate\_box`, `precise\_location`, `l\_patterns`, `is\_valid`, `score`; `.rectify(frame, output\_size)` returns the deskewed crop.
* **`Decoded`** — `detection` (the `DetectionResult`), `text` (payload), `codewords`, `matrix` (the sampled module grid).
* **`PreciseLocation`** — `vertices`, `center`, `angle`, `size`; `.ordered\_vertices()` returns integer corner points.

### Configuration objects

`DetectorConfig` groups the per-stage configs: `ExtractionConfig`, `ValidatorConfig`, `LFinderConfig`, `DashedBorderConfig`, `BorderFitterConfig`. `DecoderConfig` controls grid estimation and sampling (`output\_size`, `valid\_sizes`, `smoothing`, `estimator\_margin`). All are plain dataclasses with sensible defaults — override only what you need.

## Limitations

* Detection is sensitive to parameters; a single config will not handle all images (see *Per-image configuration*).

## Development

```bash
pip install -e ".\[test]"
pytest
```

The suite covers decode correctness against known payloads (including reflectance-inverted codes), localization across all sample images, input validation, and pipeline invariants. RANSAC is seeded for reproducibility. Any known, unfixed gaps are tracked as strict `xfail` so they stay visible and the suite alerts if a gap is ever closed.

## Project structure

```
matrixvision/
├── api.py            # public functions: detect, decode, detect\_and\_decode
├── config.py         # configuration dataclasses
├── data.py           # result/DTO types
├── debug.py          # DebugSink protocol + NullSink / CvDebugSink
├── viz.py            # drawing helpers
├── detector/         # detection pipeline and its stages
└── decoder/          # grid estimation, sampling, decoding
```
