Image Features Extractors for `nocaps`
======================================

This repository is a collection of scripts and Jupyter notebooks for extracting bottom-up image features required by the baseline models for `nocaps`.

* [Terminology](#terminology)
* [Setup Instructions](#setup-instructions)

Pre-trained weights and some parts of this codebase are adopted from [@peteanderson80/bottom-up-attention](https://www.github.com/peteanderson80/bottom-up-attention) and [@tensorflow/models](https://www.github.com/tensorflow/models).  
If you find this code useful, please consider citing our paper and these works. Bibtex available in [CITATION.md](CITATION.md).


Terminology
-----------

We have two baselines (Table 2 of our paper) using two different detectors for bottom-up image features and predicted boxes.

**Baselines:**

1. CBS - Constrained Beam Search (applied to UpDown Captioner).
2. NBT - Neural Baby Talk (with and without Constrained Beam Search).  

**Detectors:**

1. `OI Detector` - trained using Open Images v4 split, from the Tensorflow model zoo.
2. `VG Detector` - trained using Visual Genome, from (Anderson et al. CVPR 2017).  

`VG Detector` is used as a source of bottom-up image features (2048-dimensional vectors) for both the baselines (and their ablations). `OI Detector` is used as a source of CBS constraints and NBT visual word candidates. Refer our paper for further details.


Setup Instructions
------------------

We recommend using Docker for setting up our codebase. We provide two separate dockerfiles, one each for both of our detectors. We do so to avoid bulky and cluttered docker images due to separate dependencies of Caffe and Tensorflow. These steps assume that Docker is already installed. Install [nvidia-docker](https://www.github.com/nvidia/nvidia-docker), which enables usage of GPUs from inside a container.

**Note:** Follow these two steps as per your needs. For example, if you wish to use only the `OI Detector`, you need not build the docker image for `VG Detector`.

### For using `OI Detector`

1. Download [pre-trained model]() and place it in `models` sub-directory.

2. Build docker image:

    cd docker
    docker build --file oi_dockerfile --tag oi_image .

### For using `VG Detector`

1. Download [pre-trained model](https://storage.googleapis.com/bottom-up-attention/resnet101_faster_rcnn_final.caffemodel) and place it under `models/vg_faster_rcnn_end2end` sub-directory.

2. Build docker image:

    cd docker
    docker build --file vg_dockerfile --tag vg_image .
