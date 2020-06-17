import argparse
import tensorflow as tf
import os
import sys
import numpy as np
import yaml
from tqdm import tqdm

from anchor import generate_default_boxes
from box_utils import decode, compute_nms
from PIL import Image
from tensorflow.keras import models


parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='ssd300')
parser.add_argument('--num-examples', default=-1, type=int)
parser.add_argument('--gpu-id', default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

num_classes = 21
batch_size = 1

def transform_center_to_corner(boxes):
    """ Transform boxes of format (cx, cy, w, h)
        to format (xmin, ymin, xmax, ymax)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    corner_box = tf.concat([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2], axis=-1)

    return corner_box

def decode(default_boxes, locs, variance=[0.1, 0.2]):
    """ Decode regression values back to coordinates
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        locs: tensor (batch_size, num_default, 4)
              of format (cx, cy, w, h)
        variance: variance for center point and size
    Returns:
        boxes: tensor (num_default, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    locs = tf.concat([
        locs[..., :2] * variance[0] *
        default_boxes[:, 2:] + default_boxes[:, :2],
        tf.math.exp(locs[..., 2:] * variance[1]) * default_boxes[:, 2:]], axis=-1)
    boxes = transform_center_to_corner(locs)

    return boxes


def predict(confs, locs, thresh, default_boxes):

    confs = tf.squeeze(confs, 0)
    locs = tf.squeeze(locs, 0)

    confs = tf.math.softmax(confs, axis=-1)
    boxes = decode(default_boxes, locs)

    out_boxes = []
    out_labels = []
    out_scores = []

    print('confs shape =', np.shape(confs))
    print('confs max =', np.amax(confs))
    print('confs min =', np.amin(confs))
    print('confs max index =', np.where(np.array(confs) == np.array(confs).max()))

    for c in range(1, num_classes):
        cls_scores = confs[:, c]

        score_idx = cls_scores > thresh
        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]

        nms_idx = compute_nms(cls_boxes, cls_scores, 0.45, 200)
        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)
        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()

    return boxes, classes, scores


if __name__ == '__main__':
    with open('./config.yml') as f:
        cfg = yaml.load(f)
    config = cfg[args.arch.upper()]
    default_boxes = generate_default_boxes(config)

    #ssd_path = 'D:/PycharmProjects/ssd-tf2/models/ssd - Copy.h5'
    #ssd_path = 'D:/PycharmProjects/ssd-tf2/models/ssd - Copy (2).h5'
    ssd_path = 'D:/PycharmProjects/ssd-tf2/models/ssd.h5'
    ssd = models.load_model(ssd_path)
    ssd.load_weights(ssd_path)

    img = Image.open("dataset/bird_images-PascalVOC-export/JPEGImages/bluebird2.png")
    img = np.array(img.resize((300, 300)))
    img = np.expand_dims(img, 0)

    confs, locs = ssd.predict(img)

    print('confs shape =', np.shape(confs))
    print('confs max =', np.amax(confs))
    print('confs min =', np.amin(confs))
    print('confs max index =', np.where(confs==confs.max()))
    print(confs[0,7818,0])

    thresh = 0.5

    boxes, classes, scores = predict(confs, locs, thresh, default_boxes)

    print('classes =', classes)
