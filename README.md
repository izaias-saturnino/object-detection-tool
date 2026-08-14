# Object Detection Tool

A microscopy image processing and object detection pipeline for nanoparticle detection using Ultralytics YOLO with oriented bounding boxes (OBBs).

The tool provides an end-to-end workflow for preprocessing microscopy images, generating object annotations from diameter markings, preparing images for detection, training YOLO models, running inference, and post-processing detected objects.

The project was developed for nanoparticle detection in microscopy imagery, with support for both scanning electron microscopy (SEM) and transmission electron microscopy (TEM) datasets.

## Overview

The pipeline is designed around microscopy images in which nanoparticles are annotated using red diameter lines. These annotations can be automatically extracted and converted into YOLO-compatible labels.

The main workflow of this tool is:

1. Preprocess the input microscopy images.
2. Detect and extract the red diameter annotations.
3. Generate YOLO oriented bounding-box labels.
4. Normalize object annotations.
5. Resize or split images according to the particle sizes in the dataset.
6. Run a YOLO OBB model for detection.
7. Restore detections from images that were split during preprocessing.
8. Save detection results, including object coordinates and identifiers.

The same pipeline can also be used in training mode to train a new YOLO OBB model.

## Features

* Microscopy image preprocessing.
* Automatic removal of image regions below the first fully black row.
* Extraction of red diameter annotations from microscopy images.
* Automatic generation of YOLO labels.
* Oriented bounding-box (OBB) annotation generation.
* Automatic image resizing based on particle-size statistics.
* Image splitting for objects that are too small relative to the desired detection scale.
* Restoration of detections from split images.
* Oriented IoU calculation.
* Oriented non-maximum suppression (NMS).
* YOLO model training using Ultralytics.
* YOLO OBB inference.
* Automatic archival of YOLO inference runs.
* Export of detections to CSV.
* Support for SEM and TEM datasets.
* Persistent configuration through a local `config.pkl` file.

## Requirements

The project uses Python and the following main libraries:

* [Ultralytics](https://docs.ultralytics.com/)
* OpenCV
* NumPy
* scikit-learn

A Python environment with a compatible version of PyTorch is also required by Ultralytics.

For example:

```bash
pip install ultralytics opencv-python numpy scikit-learn
```

Python 3.10+ is recommended.

## Repository Structure

```text
object-detection-tool/
├── datasets/
│   └── ...
├── models/
│   └── ...
├── config.py
├── data_SEM.yaml
├── data_TEM.yaml
├── detection_post_processing.py
├── image_management.py
├── image_pre_processing.py
├── main.py
├── object-detection-tool.sh
├── object_resizer.py
├── run_models.py
├── train_model.py
├── yolo11n-obb.pt
├── yolo11n-obb_test.pt_20250218-072546.pt
├── yolov8n-obb.pt
└── yolov8n_black_particles-obb_phase_1.pt
```

The repository also contains dataset and model files used by the project.

## Main Components

### `main.py`

The main entry point for the complete pipeline.

It:

* Loads the persistent configuration.
* Preprocesses the raw images.
* Generates OBB labels.
* Runs detection or training depending on the configured mode.
* Saves detection results to CSV when enabled.
* Removes temporary processing directories after execution.
* Updates the persistent configuration.

The default mode is detection.

```bash
python main.py
```

The shell script provided by the repository runs the same entry point:

```bash
./object-detection-tool.sh
```

On systems where the shell script is not directly executable, the equivalent command is:

```bash
python ./main.py
```

## Configuration

Configuration is managed by `config.py`.

If a `config.pkl` file does not already exist, default values are created and subsequently saved to the file.

Important configuration parameters include:

| Parameter            | Default                                  | Description                                    |
| -------------------- | ---------------------------------------- | ---------------------------------------------- |
| `mode`               | `detect`                                 | Selects detection or training mode             |
| `raw_data`           | `raw_data`                               | Directory containing input images              |
| `temp_data`          | `temp_data`                              | Temporary preprocessing directory              |
| `clean_data`         | `clean_data`                             | Processed images and generated labels          |
| `detection_data`     | `detection_data`                         | Images prepared for model inference            |
| `results_data`       | `results_data`                           | Detection result output directory              |
| `default_model`      | `yolo11n-obb_test.pt_20250218-072546.pt` | Default model used for detection               |
| `default_base_model` | `yolo11n-obb.pt`                         | Base model used for training                   |
| `default_yaml`       | `data_TEM.yaml`                          | Dataset configuration used for training        |
| `max_epochs`         | `5000`                                   | Maximum number of training epochs              |
| `patience`           | `100`                                    | Early-stopping patience                        |
| `save_period`        | `10`                                     | Training checkpoint interval                   |
| `models_path`        | `models`                                 | Directory where trained models are stored      |
| `save_csv`           | `True`                                   | Save detections to CSV                         |
| `write_ids`          | `True`                                   | Include object identifiers in exported results |

These values can be modified through the configuration system or by creating/updating the generated `config.pkl`.

## Input Images and Annotation Generation

The annotation-generation pipeline assumes a specific microscopy image convention.

For an image such as:

```text
image001.png
```

the corresponding image containing diameter annotations should be named:

```text
image001xxxx.png
```

The `xxxx` suffix identifies images containing the red diameter markings.

The tool:

1. Finds images ending in `xxxx`.
2. Finds the corresponding image without the `xxxx` suffix.
3. Extracts red pixels from the annotated image.
4. Finds contours corresponding to the diameter markings.
5. Calculates the center and size of each annotation.
6. Converts the annotation into a normalized YOLO label.
7. Writes the resulting label alongside the corresponding unannotated image.

The red annotations are detected using an RGB threshold designed for the red diameter markings.

## Oriented Bounding Boxes

The current pipeline is primarily designed for oriented bounding-box detection.

When `oriented_bb=True`, each generated YOLO label contains:

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

where the four coordinate pairs describe the corners of the oriented bounding box.

The project currently uses a single object class:

```yaml
names:
  0: 'particle'

nc: 1
```

The generated labels therefore use class `0` for every detected particle.

The current annotation-generation procedure initially extracts the diameter from the red marking and represents the particle using a square OBB corresponding to that diameter.

## Dataset Configuration

Two dataset configuration files are included:

* `data_SEM.yaml`
* `data_TEM.yaml`

The SEM configuration expects:

```text
data_SEM/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

The TEM configuration follows the same structure:

```text
data_TEM/
├── images/
│   ├── train/
│   └── val/
```

with corresponding labels.

Both configurations currently define one class:

```yaml
names:
  0: 'particle'

nc: 1
```

## Image Preprocessing

`image_pre_processing.py` contains the preprocessing and annotation-generation routines.

The standard preprocessing step loads each image and removes the portion of the image below the first completely black row. This is intended to remove unwanted regions present in the microscopy images.

The preprocessing stage also maintains image metadata so that previously processed images can be identified and skipped.

The main functions are:

* `pre_process_image()`
* `pre_process_images()`
* `generate_labels()`

The unused `clean_img()` routine provides additional gradient-based image processing but is not part of the default pipeline.

## Object-Size-Based Resizing

`object_resizer.py` prepares images for inference based on the size of the annotated particles.

The tool reads OBB labels and calculates object dimensions and areas. It then derives particle-size statistics such as:

* Mean
* Median
* Standard deviation
* Variance
* Minimum
* Maximum
* Number of objects

The default resizing statistic is the median particle size.

The image is then resized relative to a target particle size.

Depending on the resulting scale factor, the tool either:

* Keeps the original image size.
* Enlarges the image by adding white space.
* Splits the image into smaller overlapping regions.

This allows particles of substantially different sizes to be presented to the detection model at a more appropriate scale.

When an image is split, the corresponding OBB coordinates are transformed to the coordinates of each generated image.

## Detection

Detection is implemented in `run_models.py`.

Before inference, the input images can be resized or split according to the object-size statistics associated with the model.

The configured YOLO model is then run with the specified confidence threshold.

By default, the main pipeline uses:

```text
confidence = 0.5
```

The model outputs are subsequently passed to the post-processing pipeline.

YOLO inference runs are automatically moved into timestamped directories to avoid overwriting previous results.

## Detection Post-Processing

`detection_post_processing.py` contains utilities for working with oriented detections.

The module includes functionality for:

* Converting OBB representations.
* Computing oriented rectangle areas.
* Computing oriented intersections.
* Computing oriented IoU.
* Grouping OBBs spatially using a grid.
* Applying oriented NMS.
* Restoring detections from split images.
* Converting detection coordinates back to the original image coordinate system.
* Exporting detection results.

When an image is divided into multiple overlapping regions, detections are first produced independently on those regions. The post-processing stage translates their coordinates back to the original image and combines the detections.

## Training

Training is implemented in `train_model.py` and can be enabled through the configuration.

Set:

```python
mode = "train"
```

The main pipeline then trains a new model using the configured base model and dataset YAML.

The current training configuration uses Ultralytics YOLO with several augmentations:

```text
Rotation: 180 degrees
Copy-paste: 0.5
MixUp: 0.5
Vertical flip: 0.5
Multi-scale training: enabled
```

The trained model is saved in the configured `models` directory with a timestamped filename.

For example:

```text
models/
└── yolo11n_particles-obbYYYYMMDD-HHMMSS_YYYYMMDD-HHMMSS.pt
```

The exact generated filename depends on the training run timestamp.

## Included Models

The repository currently contains several YOLO OBB models, including:

```text
yolo11n-obb.pt
yolo11n-obb_test.pt_20250218-072546.pt
yolov8n-obb.pt
yolov8n_black_particles-obb_phase_1.pt
```

These include base OBB models as well as models trained during the development of the nanoparticle detection pipeline.

## Typical Workflow

### Detection

Place the microscopy images in the configured `raw_data` directory and run:

```bash
python main.py
```

The default pipeline performs:

```text
raw images
    ↓
image preprocessing
    ↓
annotation extraction
    ↓
YOLO OBB label generation
    ↓
object-size-based resizing/splitting
    ↓
YOLO OBB inference
    ↓
detection restoration
    ↓
post-processing
    ↓
CSV/results
```

Temporary processing directories are removed by the main pipeline after execution.

### Training

Change the configured mode to:

```python
mode = "train"
```

and run:

```bash
python main.py
```

The pipeline will preprocess the images, generate OBB labels, and train a new YOLO model using the configured dataset YAML and base model.

## Research Context

This repository is part of a broader project on automated nanoparticle detection and measurement in microscopy images.

The related [`yolo-nanoparticle`](https://github.com/izaias-saturnino/yolo-nanoparticle) repository contains the more focused experimental training and benchmarking code for YOLO-based nanoparticle detection.

This repository instead provides a more integrated object-detection tool, combining image preparation, annotation extraction, object-size normalization, detection, and post-processing into a single pipeline.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.

## References

* [Ultralytics YOLO documentation](https://docs.ultralytics.com/)
* [YOLOv11 documentation](https://docs.ultralytics.com/models/yolov11/)
* [`yolo-nanoparticle`](https://github.com/izaias-saturnino/yolo-nanoparticle) — related nanoparticle detection research repository.
