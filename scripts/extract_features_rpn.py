import argparse
import json
import os
from typing import List, Tuple

import caffe
import cv2
import h5py
import numpy as np
import pprint
from tqdm import tqdm

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# fmt: off
parser = argparse.ArgumentParser("""
    Extract bottom-up image features from a detector. Use the detector's RPN for box proposals,
    Save boxes, predicted classes, features and their prediction confidence, in an output H5 file.
""")
# fmt: on

parser.add_argument(
    "--prototxt",
    default="models/vg_faster_rcnn_end2end/test_rpn.prototxt",
    help="Prototxt file defining the network architecture.",
)
parser.add_argument(
    "--caffemodel",
    default="models/vg_faster_rcnn_end2end/resnet101_faster_rcnn_final.caffemodel",
    help="Caffemodel file containing weights of pretrained model.",
)
parser.add_argument(
    "--images", help="Path to a directory containing images for a particular split."
)
parser.add_argument(
    "--annotations",
    help="Path to annotations JSON file (in COCO format) containing image info.",
)
parser.add_argument(
    "--output",
    default="features.h5",
    help="Path to save the output H5 file with image features.",
)
parser.add_argument("--gpu-id", default=0, type=int, help="Which GPU ID to use.")

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36.
MIN_BOXES = 10
MAX_BOXES = 100

# Maximum size of the largest side of the image.
MAX_DIMENSION = 1024


def _image_ids(annotations_path: str) -> List[Tuple[int, str]]:
    r"""
    Given an annotation file in COCO format, return a list of ``(image_id, filename)`` tuples.
    """

    image_annotations = json.load(open(annotations_path))["images"]
    image_ids = [(im["id"], im["file_name"]) for im in image_annotations]

    image_ids = sorted(image_ids, key=lambda k: k[0])
    return image_ids


def resize_image(image: np.ndarray) -> np.ndarray:

    height, width, _ = image.shape
    largest_side_index = np.argmax([height, width])

    if image.shape[largest_side_index] > MAX_DIMENSION:

        factor = image.shape[largest_side_index] / MAX_DIMENSION
        new_dimensions = [0, 0]
        new_dimensions[np.mod(largest_side_index + 1, 2)] = (
            image.shape[np.mod(largest_size_index + 1, 2)] / factor
        )
        new_dimensions[largest_side_index] = MAX_DIMENSION

        image = cv2.resize(image, (new_dimensions[1], new_dimensions[0]))

    return image


def get_detections_from_im(
    net, image_file: str, image_id: int, conf_thresh: float = 0.2
):
    # shape: (height, width, channels)
    image = cv2.imread(image_file)
    height, width, _ = image.shape

    image = resize_image(image)

    scores, boxes, attr_scores, rel_scores = im_detect(net, image)

    # Keep the original boxes, don't worry about the regresssion bbox outputs.
    rois = net.blobs["rois"].data.copy()
    # Unscale back to raw image space.
    blobs, image_scales = _get_blobs(image, None)

    cls_boxes = rois[:, 1:5] / image_scales[0]
    cls_prob = net.blobs["cls_prob"].data
    features = net.blobs["pool5_flat"].data

    # Keep only the best detections.
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(
            cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep]
        )

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

    return {
        "image_id": image_id,
        "height": height,
        "width": width,
        "boxes": cls_boxes[keep_boxes],
        "features": features[keep_boxes],
    }


if __name__ == "__main__":

    args = parser.parse_args()
    print("Using config:")
    pprint.pprint(cfg)

    image_ids = _image_ids(args.annotations)

    output_h5 = h5py.File(args.output, "w")
    dt = h5py.special_dtype(vlen=np.dtype("float32"))

    image_id_dset = output_h5.create_dataset("image_id", (len(image_ids),), np.int64)
    height_dset = output_h5.create_dataset("height", (len(image_ids),), np.float32)
    width_dset = output_h5.create_dataset("width", (len(image_ids),), np.float32)
    boxes_dset = output_h5.create_dataset("boxes", (len(image_ids),), dt)
    features_dset = output_h5.create_dataset("features", (len(image_ids),), dt)

    caffe.init_log()
    caffe.log("Using device {}".format(args.gpu_id))
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    net = caffe.Net(args.prototxt, caffe.TEST, weights=args.caffemodel)

    for index, (image_id, image_file) in enumerate(tqdm(image_ids)):
        features_dict = get_detections_from_im(
            net, os.path.join(args.images, image_file), image_id
        )

        image_id_dset[index] = features_dict["image_id"]
        height_dset[index] = features_dict["height"]
        width_dset[index] = features_dict["width"]
        boxes_dset[index] = features_dict["boxes"].reshape(-1)
        features_dset[index] = features_dict["features"].reshape(-1)

    output_h5.close()
