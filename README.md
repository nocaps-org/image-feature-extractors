Image Features Extractors for `nocaps`
======================================

This repository is a collection of scripts and Jupyter notebooks for extracting bottom-up image features required by the baseline models for `nocaps`.

* [Terminology](#terminology)
* [Setup Instructions](#setup-instructions)
* [Extract Boxes from `OI Detector`](#extract-boxes-from-oi-detector)

Pre-trained weights and some parts of this codebase are adopted from [@peteanderson80/bottom-up-attention](https://www.github.com/peteanderson80/bottom-up-attention) and [@tensorflow/models](https://www.github.com/tensorflow/models).  
If you find this code useful, please consider citing our paper and these works. Bibtex available in [CITATION.md](CITATION.md).


Terminology
-----------

We have two baselines (Table 2 of our paper) using two different detectors for bottom-up image features and predicted boxes.

**Baselines:**

1. CBS - Constrained Beam Search (applied to UpDown Captioner).
2. NBT - Neural Baby Talk (with and without Constrained Beam Search).  

**Detectors:**

1. `OI Detector` - trained using Open Images v4 split, from the [Tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
2. `VG Detector` - trained using Visual Genome, from [@peteanderson80/bottom-up-attention](https://www.github.com/peteanderson80/bottom-up-attention).  

`VG Detector` is used as a source of bottom-up image features (2048-dimensional vectors) for both the baselines (and their ablations). `OI Detector` is used as a source of CBS constraints and NBT visual word candidates. Refer our paper for further details.


Setup Instructions
------------------

Seeting up this codebase requires Docker, so install it first. We provide two separate dockerfiles, one each for both of our detectors. Also, install [nvidia-docker](https://www.github.com/nvidia/nvidia-docker), which enables usage of GPUs from inside a container.

**Note:** Follow these two steps as per your needs. For example, if you wish to use only the `OI Detector`, you need not build the docker image for `VG Detector`.

### For using `OI Detector`

1. Download [pre-trained model]() and place it in `models` sub-directory.

2. Build docker image:

```shell
cd docker
docker build --file oi_dockerfile --tag oi_image .
```

### For using `VG Detector`

1. Download [pre-trained model](https://storage.googleapis.com/bottom-up-attention/resnet101_faster_rcnn_final.caffemodel) and place it under `models/vg_faster_rcnn_end2end` sub-directory.

2. Build docker image:

```shell
cd docker
docker build --file vg_dockerfile --tag vg_image .
```


Extract Boxes from `OI Detector`
--------------------------------

We use `OI Detector` as a source of bounding boxes (and associated class labels) for CBS constraints and candidate grounding regions for NBT (rows 2-4, Table 2 of our paper). We do not use the bottom-up image features extracted from this detector in any experiments.

Launch the docker image in a container, make sure to attach project root directory as well as directories containing dataset split as volumes.

```shell
nvidia-docker run -it \
    --name oi_container \
    -v $PWD:/workspace \                   # attach project root as volume
    -v /path/to/nocaps:/datasets/nocaps \  # omit this if using only coco
    -v /path/to/coco:/datasets/coco \      # omit this if using only nocaps
    -p 8880:8880                           # port forward for accessing jupyter notebook/lab
    oi_image
    /bin/bash
```

Inside the container environment, extract boxes with this command (example for `nocaps` val):

```shell
python3 scripts/extract_boxes_oi.py \
    --graph models/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018/frozen_inference_graph.pb \
    --images /datasets/nocaps/images/val \
    --annotations /datasets/nocaps/annotations/nocaps_val_image_info.json \
    --output /outputs/nocaps_val_detections.json
```

This creates a JSON file with bounding boxes in COCO format (of instance annotations). These detections can be used to generate CBS constraints, or the bounding boxes can be used to extract 2048-d bottom-up features from `VG Detector` for NBT candidate grounding regions.
