import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

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
    "--force-boxes",
    deault=None,
    help="Path to JSON file (in COCO format) containing external boxes (instead of RPN).",
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
    Given path to an annotation file in COCO format, return ``(image_id, filename)`` tuples.

    Parameters
    ----------
    annotations_path: str
        Path to an annotation file in COCO format. Must contain "images" key.

    Returns
    -------
    List[Tuple[int, str]]
        List of ``(image_id, filename)`` tuples.
    """

    image_annotations = json.load(open(annotations_path))["images"]
    image_ids = [(im["id"], im["file_name"]) for im in image_annotations]

    image_ids = sorted(image_ids, key=lambda k: k[0])
    return image_ids


def resize_larger_edge(
    image: np.ndarray, dimension: int, force_boxes: Optional[np.ndarray] = None
) -> np.ndarray:
    r"""
    Resize larger edge of the image to ``dimension`` while preserving the aspect ratio.

    Parameters
    ----------
    image: np.ndarray
        An array of shape ``(height, width, channels)`` with values in [0, 255] representing
        an RGB image.
    dimension: np.ndarray
        The dimension of larger edge of image.
    force_boxes: np.ndarray, optional (default = None)
        If the boxes are provided externally instead of usng detector's RPN.

    Returns
    -------
    np.ndarray
        Resized RGB image.
    """

    height, width, _ = image.shape
    largest_side_index = np.argmax([height, width])

    if image.shape[largest_side_index] > MAX_DIMENSION:

        if force_boxes is not None:
            # Normalize boxes between [0, 1] before resizing.
            # Boxes are of form [X1, Y1, X2, Y2]
            force_boxes[:, 0] /= float(width)
            force_boxes[:, 2] /= float(width)
            force_boxes[:, 1] /= float(height)
            force_boxes[:, 3] /= float(height)

        factor = image.shape[largest_side_index] / MAX_DIMENSION
        new_dimensions = [0, 0]
        new_dimensions[np.mod(largest_side_index + 1, 2)] = (
            image.shape[np.mod(largest_size_index + 1, 2)] / factor
        )
        new_dimensions[largest_side_index] = MAX_DIMENSION

        image = cv2.resize(image, (new_dimensions[1], new_dimensions[0]))
        height, width, _ = image.shape

        if force_boxes is not None:
            # Unnormalize the boxes with new width and height.
            force_boxes[:, 0] *= float(width)
            force_boxes[:, 2] *= float(width)
            force_boxes[:, 1] *= float(height)
            force_boxes[:, 3] *= float(height)

    return image, force_boxes


def get_detections_from_im(
    net,
    image_file: str,
    image_id: int,
    force_boxes: Optional[np.ndarray] = None,
    conf_thresh: float = 0.2,
):
    # shape: (height, width, channels)
    # Record original height and width of the image (before resizing).
    image = cv2.imread(image_file)
    height, width, _ = image.shape

    image, force_boxes = resize_larger_edge(image, MAX_DIMENSION, force_boxes)

    scores, boxes, _, _ = im_detect(
        net, image, boxes=force_boxes, force_boxes=(force_boxes is not None)
    )
    # Keep the original boxes, don't worry about the regression bbox outputs.
    rois = net.blobs["rois"].data.copy()
    # Unscale back to raw image space.
    blobs, image_scales = _get_blobs(image, None)

    extracted_boxes = rois[:, 1:5] / image_scales[0]
    extracted_features = net.blobs["pool5_flat"].data

    if force_boxes is None:
        # Keep only the best detections above a confidence threshold if using RPN.
        # For each box, these are confidence scores for all classes.
        # shape: (num_boxes, num_classes)
        cls_prob = net.blobs["cls_prob"].data

        # Confidence threshold for at least one class should be above a threshold.
        max_conf = np.zeros((rois.shape[0]))
        for cls_ind in range(1, cls_prob.shape[1]):
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((extracted_boxes, cls_scores[:, np.newaxis])).astype(
                np.float32
            )
            keep = np.array(nms(dets, cfg.TEST.NMS))
            max_conf[keep] = np.where(
                cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep]
            )

        keep_boxes = np.where(max_conf >= conf_thresh)[0]
        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

        # When we use RPN and not external boxes, we are most likely extracting class agnostic
        # bottom-up features. So don't extract "classes".
        extracted_boxes = extracted_boxes[keep_boxes]
        extracted_features = extracted_features[keep_boxes]
        extracted_scores = max_conf[keep_boxes]

    output_dict = {
        "image_id": image_id,
        "height": height,
        "width": width,
        "boxes": extracted_boxes,
        "features": extracted_features,
    }

    # If using RPN, include confidence score. If boxes are provided externally, then ignore the
    # confidence score from this detector.
    if force_boxes is None:
        output_dict["scores"] = extracted_scores


if __name__ == "__main__":

    _A = parser.parse_args()
    print("Using config:")
    pprint.pprint(cfg)

    # List of tuples of image IDs and their file names.
    image_ids = _image_ids(_A.annotations)

    output_h5 = h5py.File(_A.output, "w")
    dt = h5py.special_dtype(vlen=np.float32)

    image_id_dset = output_h5.create_dataset("image_id", (len(image_ids),), np.int64)
    height_dset = output_h5.create_dataset("height", (len(image_ids),), np.float32)
    width_dset = output_h5.create_dataset("width", (len(image_ids),), np.float32)
    boxes_dset = output_h5.create_dataset("boxes", (len(image_ids),), dt)
    features_dset = output_h5.create_dataset("features", (len(image_ids),), dt)
    scores_dset = output_h5.create_dataset("scores", (len(image_ids),), dt)

    if _A.force_boxes is not None:
        # Externally provided boxes in COCO format.
        force_boxes_json = json.load(open(_A.force_boxes))["annotations"]

        # Keep a map of image ID to force boxes.
        force_boxes_map: Dict[str, Any] = {}
        for annotation in force_boxes_json:
            if annotation["image_id"] not in force_boxes:
                force_boxes_map[annotation["image_id"]] = [annotation]
            else:
                force_boxes_map[annotation["image_id"]].append(annotation)

        # Make an H5 dataset to also store predicted classes if external boxes are provided.
        classes_dset = output_h5.create_dataset(
            "classes",
            (len(image_ids),),
            h5py.special_dtype(vlen=np.uint32)
        )

    caffe.init_log()
    caffe.log("Using device {}".format(_A.gpu_id))
    caffe.set_mode_gpu()
    caffe.set_device(_A.gpu_id)

    net = caffe.Net(_A.prototxt, caffe.TEST, weights=_A.caffemodel)

    for index, (image_id, image_file) in enumerate(tqdm(image_ids)):

        if _A.force_boxes is not None:
            # Get force_boxes if provided through args.
            force_boxes_annotations = force_boxes_map[image_id]
            force_boxes = np.asarray(
                [a["bbox"] for a in force_boxes_annotations], dtype=np.float32
            )
        else:
            force_boxes = None

        features_dict = get_detections_from_im(
            net, os.path.join(_A.images, image_file), image_id, force_boxes=force_boxes
        )

        image_id_dset[index] = features_dict["image_id"]
        height_dset[index] = features_dict["height"]
        width_dset[index] = features_dict["width"]
        boxes_dset[index] = features_dict["boxes"].reshape(-1)
        features_dset[index] = features_dict["features"].reshape(-1)

        if _A.force_boxes is not None:
            # Save classes and confidence scores in output H5.
            classes_dset[index] = np.asarray([a["category_id"] for a in force_boxes_annotations])
            scores_dset[index] = np.asarray([a.get("score", 1.0) for a in force_boxes_annotations])
        else:
            # Save confidence scores from the detector if using RPN.
            scores_dset[index] = features_dict["scores"].reshape(-1)

    output_h5.close()
