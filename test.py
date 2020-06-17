import argparse
import tensorflow as tf
import os
import sys
import numpy as np
import yaml
from tqdm import tqdm

from anchor import generate_default_boxes
from box_utils import decode, compute_nms
from voc_data import create_batch_generator
from image_utils import ImageVisualizer
from losses import create_losses
from get_model import get_VGG16_SSD_300, get_mobilenet_SSD
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='./dataset/bird_images-PascalVOC-export')
parser.add_argument('--data-year', default='2007')
parser.add_argument('--arch', default='ssd300')
parser.add_argument('--num-examples', default=-1, type=int)
parser.add_argument('--gpu-id', default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

num_classes = 21
batch_size = 1


def predict(img, default_boxes):
    confs, locs = ssd(img)

    confs = tf.squeeze(confs, 0)
    locs = tf.squeeze(locs, 0)

    confs = tf.math.softmax(confs, axis=-1)
    classes = tf.math.argmax(confs, axis=-1)
    scores = tf.math.reduce_max(confs, axis=-1)

    boxes = decode(default_boxes, locs)

    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, num_classes):
        cls_scores = confs[:, c]

        score_idx = cls_scores > 0.6
        # cls_boxes = tf.boolean_mask(boxes, score_idx)
        # cls_scores = tf.boolean_mask(cls_scores, score_idx)
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

    try:
        config = cfg[args.arch.upper()]
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format(args.arch))

    default_boxes = generate_default_boxes(config)

    batch_generator, info = create_batch_generator(
        args.data_dir, default_boxes,
        config['image_size'],
        batch_size, args.num_examples, mode='test')

    try:
        ssd = get_mobilenet_SSD(image_size=(300,300,3), num_classes=num_classes)
    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()

    os.makedirs('outputs/images', exist_ok=True)
    os.makedirs('outputs/detects', exist_ok=True)
    visualizer = ImageVisualizer(info['idx_to_name'], save_dir='outputs/images')

    for i, (filename, imgs, gt_confs, gt_locs) in enumerate(
        tqdm(batch_generator, total=info['length'],
             desc='Testing...', unit='images')):
        boxes, classes, scores = predict(imgs, default_boxes)
        filename = filename.numpy()[0].decode()
        original_image = Image.open(
            os.path.join(info['image_dir'], '{}.jpg'.format(filename)))
        boxes *= original_image.size * 2
        visualizer.save_image(
            original_image, boxes, classes, '{}.jpg'.format(filename))

        log_file = os.path.join('outputs/detects', '{}.txt')

        for cls, box, score in zip(classes, boxes, scores):
            cls_name = info['idx_to_name'][cls - 1]
            with open(log_file.format(cls_name), 'a') as f:
                f.write('{} {} {} {} {} {}\n'.format(
                    filename,
                    score,
                    *[coord for coord in box]))
