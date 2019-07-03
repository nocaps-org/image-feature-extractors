#!/usr/bin/env python

"""
Generate bottom-up attention features as a tsv file.
Modify the load_image_ids script as necessary for your data location.
"""


# Example:
# ./tools/extract_features.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --out test2014_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_test2014
# ./tools/extract_features_h5.py --gpu 7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_val2017

import argparse
import base64
import csv
import pprint
import os, sys
import json

import caffe
import h5py
import numpy as np
import cv2
from tqdm import tqdm


csv.field_size_limit(sys.maxsize)
sys.path.append("/app/lib")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms


MIN_BOXES = 18
MAX_BOXES = 18


parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                    default='0', type=str)
parser.add_argument('--def', dest='prototxt',
                    help='prototxt file defining the network',
                    default=None, type=str)
parser.add_argument('--net', dest='caffemodel',
                    help='model to use',
                    default=None, type=str)
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional config file', default=None, type=str)
parser.add_argument('--split', dest='data_split',
                    help='dataset to use',
                    default='coco_val2017', type=str)
parser.add_argument('--set', dest='set_cfgs',
                    help='set config keys', default=None,
                    nargs=argparse.REMAINDER)


def load_image_ids(split_name):
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    split = []
    if split_name == 'coco_train2017':
        with open('/srv/share/datasets/coco/annotations/instances_train2017.json') as f:
            box_info = 'coco_train2017_roi.h5'
            data = json.load(f)
            im_index = {
                im["id"]: im for im in data["images"]
            }
            with h5py.File(box_info, "r") as boxes_file:
                for image_id in boxes_file["image_id"]:
                    item = im_index[image_id]
                    filepath = os.path.join('/srv/share/datasets/coco/images/train2017', item['file_name'])
                    split.append((filepath, image_id))
    elif split_name == 'coco_val2017':
        with open('/srv/share/datasets/coco/annotations/instances_val2017.json') as f:
            box_info = 'coco_val2017_roi.h5'
            data = json.load(f)
            im_index = {
                im["id"]: im for im in data["images"]
            }
            with h5py.File(box_info, "r") as boxes_file:
                for image_id in boxes_file["image_id"]:
                    item = im_index[image_id]
                    filepath = os.path.join('/srv/share/datasets/coco/images/val2017', item['file_name'])
                    split.append((filepath, image_id))
    elif split_name == 'nocaps_val':
        with open('/srv/share/datasets/nocaps/annotations/nocaps_val_4500_image_info.json') as f:
            # box_info = '/features/nocaps_val/tf_faster_rcnn_inception_resnet_v2_atrous_oid_v4_boxes.h5'
            box_info = 'nocaps_val_gold_boxes.h5'
            data = json.load(f)
            im_index = {
                im["id"]: im for im in data["images"]
            }
            with h5py.File(box_info, "r") as boxes_file:
                for image_id in boxes_file["image_id"]:
                    item = im_index[image_id]
                    filepath = os.path.join('/srv/share/datasets/nocaps/images/val', item['file_name'])
                    split.append((filepath, image_id))
    elif split_name == 'nocaps_test':
        with open('/srv/share/datasets/nocaps/annotations/nocaps_test_10600_image_info.json') as f:
            # box_info = '/features/nocaps_test/tf_faster_rcnn_inception_resnet_v2_atrous_oid_v4_boxes.h5'
            box_info = 'nocaps_test_gold_boxes.h5'
            data = json.load(f)
            im_index = {
                im["id"]: im for im in data["images"]
            }
            with h5py.File(box_info, "r") as boxes_file:
                for image_id in boxes_file["image_id"]:
                    item = im_index[image_id]
                    filepath = os.path.join('/srv/share/datasets/nocaps/images/test', item['file_name'])
                    split.append((filepath, image_id))

    return split, box_info


def resize_image(img, boxes):
    # max dimension is 1024
    max_dim = 1024
    height = img.shape[0]
    width = img.shape[1]

    aspectRatio = float(img.shape[1]) / img.shape[0]
    idx = np.argmax([img.shape[0], img.shape[1]])
    area = img.shape[1] * img.shape[0]
    new_dim = [0, 0]

    boxes = boxes.astype(float)

    if img.shape[idx] > max_dim:

        # Boxes are of form [X1, Y1, X2, Y2]
        boxes[:, 0] /= float(width)
        boxes[:, 2] /= float(width)
        boxes[:, 1] /= float(height)
        boxes[:, 3] /= float(height)

        factor = img.shape[idx] / max_dim
        new_dim[np.mod(idx + 1, 2)] = int(img.shape[np.mod(idx + 1, 2)] / factor)
        new_dim[idx] = max_dim
        img2 = cv2.resize(img, (new_dim[1],new_dim[0]))

        new_width = img.shape[1]
        new_height = img.shape[2]

        # Boxes are of form [X1, Y1, X2, Y2]
        boxes[:, 0] *= float(new_width)
        boxes[:, 2] *= float(new_width)
        boxes[:, 1] *= float(new_height)
        boxes[:, 3] *= float(new_height)

        return img2, boxes

    return img, boxes


def get_detections_from_im(net, im_file, image_id, force_boxes, conf_thresh=0.2):

    im = cv2.imread(im_file) # shape (rows, columns, channels)
    im, force_boxes = resize_image(im, force_boxes)

    scores, boxes, attr_scores, rel_scores = im_detect(net, im, force_boxes, True)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, force_boxes)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    pool5 = net.blobs['pool5_flat'].data

    # Make features corresponding to 0 boxes as zero:
    pool5 = (((force_boxes != 0).sum(-1) != 0).reshape(18, 1) * pool5)
    return pool5


def populate_h5(gpu_id, prototxt, weights, image_ids, box_info):
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffe.TEST, weights=weights)

    im_index = {
        image_id: index for index, image_id in enumerate(box_info["image_id"][:])
    }
    for index, (im_file, image_id) in enumerate(tqdm(image_ids)):
        try:
            force_boxes = np.array(box_info["boxes"][im_index[image_id]])
            features = get_detections_from_im(net, im_file, image_id, force_boxes)
            box_info["features"][im_index[image_id]] = features
        except:
            print("Skipped: ", image_id, im_file)


if __name__ == '__main__':

    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = int(args.gpu_id)

    print('Using config:')
    pprint.pprint(cfg)

    image_ids, box_info = load_image_ids(args.data_split)

    box_info = h5py.File(box_info)
    box_info["features"] = np.zeros((len(image_ids), 18, 2048))

    caffe.init_log()
    caffe.log('Using device %s' % str(gpu_id))

    populate_h5(gpu_id, args.prototxt, args.caffemodel, image_ids, box_info)
    box_info.close()
