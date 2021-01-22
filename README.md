Image Feature Extractors for `nocaps`
=====================================

This repository is a collection of scripts and Jupyter notebooks for extracting bottom-up image features required by the baseline models for `nocaps`.

* [Terminology](#terminology)
* [Setup Instructions](#setup-instructions)
* [Extract Boxes from `OI Detector`](#extract-boxes-from-oi-detector)
* [Extract Features from `VG Detector`](#extract-features-from-vg-detector)
* [Visualize Bounding Boxes](#visualize-bounding-boxes)
* [Frequently Asked Questions](#frequently-asked-questions)

Pre-trained weights and some parts of this codebase are adopted from [@peteanderson80/bottom-up-attention][vgrepo] and [@Tensorflow Object Detection API][oirepo].  
If you find this code useful, please consider citing our paper and these works. Bibtex available in [CITATION.md](CITATION.md).


Terminology
-----------

We have two baselines (Table 2 of our paper) using two detectors for getting boxes and image features.

**Baselines:**

1. CBS - Constrained Beam Search (applied to UpDown Captioner).
2. NBT - Neural Baby Talk (with and without Constrained Beam Search).  

**Detectors:**

1. `OI Detector` - trained using Open Images v4 split, from the [Tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
2. `VG Detector` - trained using Visual Genome, from [@peteanderson80/bottom-up-attention][oirepo].  

`OI Detector` is used as a source of CBS constraints and NBT visual word candidates. `VG Detector` is used as a source of bottom-up image features (2048-dimensional vectors) for both the baselines (and their ablations). Refer our paper for further details.


Setup Instructions
------------------

Setting up this codebase requires Docker, so install it first. We provide two separate dockerfiles, one each for both of our detectors. Also, install [nvidia-docker](https://www.github.com/nvidia/nvidia-docker), which enables usage of GPUs from inside a container.

1. **Download pre-trained models.**
    - `OI Detector`: download from [here](https://www.dropbox.com/s/uoai4xqfdx96q2c/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018.tar.gz) and un-tar it in `models` sub-directory.
    - `VG Detector`: download from [here](https://www.dropbox.com/s/5xethd2nxa8qrnq/resnet101_faster_rcnn_final.caffemodel?dl=1) and place it in `models/vg_faster_rcnn_end2end` sub-directory.

2. **Build docker image** (replace `<detector>` with `oi` or `vg` as per needs). If you wish to use only the `OI Detector`, you need not build the docker image for `VG Detector`.

```shell
docker build --file Dockerfile_<detector> --tag <detector>_image .
```


Extract Boxes from `OI Detector`
--------------------------------

We use `OI Detector` as a source of bounding boxes (and associated class labels) for CBS constraints and candidate grounding regions for NBT (rows 2-4, Table 2 of our paper). We do not use the bottom-up image features extracted from this detector in any experiments.

1. Launch the docker image in a container, make sure to attach project root directory as well as directories containing dataset split as volumes.

```shell
nvidia-docker run -it \
    --name oi_container \
    -v $PWD/scripts:/workspace/scripts \   # attach scripts as volume: reflect changes inside on editing
    -v /path/to/nocaps:/datasets/nocaps \  # omit this if using only coco
    -v /path/to/coco:/datasets/coco \      # omit this if using only nocaps
    -p 8880:8880                           # port forward for accessing jupyter notebook/lab
    oi_image
    /bin/bash
```

2. Inside the container environment, extract boxes with this command (example for `nocaps` val):

```shell
python3 scripts/extract_boxes_oi.py \
    --graph models/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018/frozen_inference_graph.pb \
    --images /datasets/nocaps/images/val \
    --annotations /datasets/nocaps/annotations/nocaps_val_image_info.json \
    --output /outputs/nocaps_val_detections.json
```

3. Copy the output file from the container environment outside to the main filesystem.

```shell
docker container cp oi_container:/outputs/nocaps_val_detections.json .
```

### Output Format

The output is a JSON file with bounding boxes in COCO format (of instance annotations):

```python
{
    "categories": [ {"id": int, "name": str, "supercategory": str}, ... ],
    "images": [ {"id": int, "file_name": str, "width": int, "height": int }, ... ],
    "annotations": [ {"image_id": int, "category_id": int, "bbox": [X1, Y1, X2, Y2], "score": float } ... ]
}
```

Note that `bbox` is of the form `[X1, Y1, X2, Y2]` as opposed to `[X, Y, W, H]` in COCO format.


Extract Features from `VG Detector`
-----------------------------------

We use `VG Detector` as a source of bottom-up image features (2048-dimensional vectors) for UpDown model and NBT. Only for the candidate grounding regions of NBT, we take boxes from `OI Detector` and use them here (instead of this detector's RPN) to get features.

Launch the docker image as for `OI Detector`:

```shell
nvidia-docker run -it \
    --name vg_container \
    -v $PWD/scripts:/workspace/scripts \   # attach scripts as volume: reflect changes inside on editing
    -v /path/to/nocaps:/datasets/nocaps \  # omit this if using only coco
    -v /path/to/coco:/datasets/coco \      # omit this if using only nocaps
    -p 8880:8880                           # port forward for accessing jupyter notebook/lab
    vg_image
    /bin/bash
```

Inside the container environment, extract class-agnostic features for UpDown model and the language model of NBT with this command (example for `nocaps` val):

```shell
python scripts/extract_features_vg.py \
    --prototxt models/vg_faster_rcnn_end2end/test_rpn.prototxt \
    --caffemodel models/vg_faster_rcnn_end2end/resnet101_faster_rcnn_final.caffemodel \
    --images /datasets/nocaps/images/val \
    --annotations /datasets/nocaps/annotations/nocaps_val_4500_image_info.json \
    --output /outputs/nocaps_val_features.h5
```

To extract features from boxes provided by `OI Detector`, add/change two arguments:
    - `--prototxt models/vg_faster_rcnn_end2end/test_force_boxes.prototxt`.
    - provide path to output JSON from `OI Detector` as `--force-boxes`.

Copy the output file from the container environment outside to the main filesystem.

```shell
docker container cp vg_container:/outputs/nocaps_val_features.h5 .
```

### Output Format

The output is a stand-alone H5 file with the following fields (one row corresponds to one image):

```python
{
    "image_id": int,
    "width": int,
    "height": int,
    "num_boxes": int,
    "boxes": np.ndarray,     # shape: (num_boxes * 4, )
    "classes": np.ndarray,   # shape: (num_boxes, )
    "scores": np.ndarray,    # shape: (num_boxes, )
    "features": np.ndarray,  # shape: (num_boxes * 2048, )
}
```

- In case of class-agnostic features extracted by `VG Detector`, `classes` field is absent.
- In case of boxes provided by `OI Detector`, `classes`, `scores`, `boxes` as the same as in JSON provided through `--force-boxes` argument.


Visualize Bounding Boxes
------------------------

We provide two notebooks in `notebooks` directory to visualize boxes and object classes from a JSON file or an H5 file. These can be run directly from either of the container environments.


Frequently Asked Questions
--------------------------

1. How do I train my own detector(s)?
    - We only provide support for feature extraction and visualization (for debugging). We do not intend to add training support in this repository in the future. For training your own detector(s), use [@peteanderson80/bottom-up-detection][vgrepo] for `VG Detector` and [Tensorflow Object Detection API][oirepo] for `OI Detector`.

2. Feature extraction is slow, how can I speed it up?
    - Feature extraction for nocaps splits is reasonably fast due to a smaller split size (~5K/~10K images), COCO train2017 would take relatively longer (~118K images). Parallelizing across multiple GPUs is an alternative, but it is unfortunately not supported. Feature extraction was a one time job for our experiments, hence introducing multi-GPU support took lower priority than other things. We do welcome Pull Requests for this support!


[vgrepo]: https://www.github.com/peteanderson80/bottom-up-attention
[oirepo]: https://github.com/tensorflow/models/blob/master/research/object_detection
