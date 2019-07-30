import argparse
import json
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument(
    "--graph",
    default="models/open_images_train_tensorflow_detection_api/"
    "faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018/frozen_inference_graph.pb",
    help="Path to frozen inference graph of pre-trained detector.",
)
parser.add_argument(
    "--images", help="Path to a directory containing images for a particular split."
)
parser.add_argument(
    "--annotations",
    help="Path to annotations JSON file (in COCO format) containing image info.",
)
parser.add_argument(
    "--boxes-per-image",
    type=int,
    default=18,
    help="Number of detected bounding boxes per image.",
)
parser.add_argument(
    "--output",
    default="/outputs/detections.json",
    help="Path to save the output JSON (in COCO format) with detected boxes.",
)


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


if __name__ == "__main__":
    _A = parser.parse_args()

    # List of tuples of image IDs and their file names.
    image_ids = _image_ids(_A.annotations)

    # Populate this dict with all the detected boxes (in COCO format).
    output_coco_format = {"categories": [], "images": [], "annotations": []}

    # --------------------------------------------------------------------------------------------
    # Load Faster-RCNN frozen inference graph. Contains both, architecture definition and weights.
    rcnn_frozen_inference_graph = tf.Graph()

    with rcnn_frozen_inference_graph.as_default():
        rcnn_frozen_inference_graphdef = tf.GraphDef()

        with tf.gfile.GFile(_A.graph, "rb") as f:
            rcnn_frozen_inference_graphdef.ParseFromString(f.read())
            tf.import_graph_def(rcnn_frozen_inference_graphdef, name="")

        # Get handles to input and output tensors.
        image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")
        detection_outputs = {
            key: tf.get_default_graph().get_tensor_by_name(key + ":0")
            for key in [
                "num_detections",
                "detection_boxes",
                "detection_scores",
                "detection_classes",
            ]
        }
    # --------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------
    # Run inference on all images and save predicted boxes, classes and confidence scores.
    with rcnn_frozen_inference_graph.as_default():
        session = tf.Session()

        for image_id, image_filename in tqdm(image_ids):

            image_path = os.path.join(_A.images, image_filename)
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size
            image_ndarray = np.array(image)

            #             if len(image_ndarray.shape) == 2:
            #                 # Add RGB channel for single-channel grayscale images.
            #                 image_ndarray = np.expand_dims(image_ndarray, axis=-1)
            #                 image_ndarray = np.repeat(image_ndarray, 3, axis=-1)
            #             elif image_ndarray.shape[2] == 4:
            #                 # Drop alpha channel from RGB-A images.
            #                 image_ndarray = image_ndarray[:, :, :3]

            # Run inference on image (add batch dimension first).
            output_dict = session.run(
                detection_outputs,
                feed_dict={image_tensor: np.expand_dims(image_ndarray, 0)},
            )

            # Remove the batch dimension and cast outputs to appropriate types.
            output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
            output_dict["detection_classes"] = output_dict["detection_classes"][0]
            output_dict["detection_scores"] = output_dict["detection_scores"][0]

            # Populate the image info in COCO format. This is just for completeness of the output
            # detections JSON file.
            output_coco_format["images"].append(
                {
                    "id": image_id,
                    "file_name": image_filename,
                    "height": image_height,
                    "width": image_width,
                }
            )

            # Populate the output detections list with these detections.
            # Boxes (and corresponding classes) list is sorted by decreasing confidence score.
            for box, clss, score in zip(
                output_dict["detection_boxes"],
                output_dict["detection_classes"],
                output_dict["detection_scores"],
            ):
                if sum(box) > 0:
                    # This is not a zero-area box (padding).

                    # Boxes are of the form [Y1, X1, Y2, X2] in [0, 1].
                    # Convert to [X1, Y1, X2, Y2]. Also, un-normalize by image width and height.
                    box = [
                        box[1] * image_width,
                        box[0] * image_height,
                        box[3] * image_width,
                        box[2] * image_height,
                    ]

                    output_coco_format["annotations"].append(
                        {
                            "image_id": image_id,
                            "category_id": int(clss),
                            "bbox": [float(coordinate) for coordinate in box],
                            "score": float(score),
                        }
                    )
    # --------------------------------------------------------------------------------------------

    # Populate the (Open Images) categories field from external file, for completeness.
    # This path is relative to $PROJECT_ROOT, so make sure to run script from there.
    output_coco_format["categories"] = json.load(open("data/oi_categories.json"))

    print("Saving output detections to {}...".format(_A.output))
    json.dump(output_coco_format, open(_A.output, "w"))
