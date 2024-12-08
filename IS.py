from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tarfile
import math
import numpy as np
import tensorflow as tf
from tqdm import trange
from imageio import imread
from PIL import Image
import urllib.request

# Define image resizing function
def imresize(image, size):
    img = Image.fromarray(image)
    return img.resize(size)

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_npy_file', default=None)
parser.add_argument('--input_image_dir', default=None)
parser.add_argument('--input_image_dir_list', default=None)
parser.add_argument('--input_image_superdir', default=None)
parser.add_argument('--image_size', default=128, type=int)
parser.add_argument('--num_splits', default=3, type=int)
parser.add_argument('--tensor_layout', default='NHWC', choices=['NHWC', 'NCHW'])

IMAGE_EXTS = ['.png', '.jpg', '.jpeg']

# Load images from a directory
def load_images(image_dir, image_size):
    print('Loading images from ', image_dir)
    images = []
    for fn in os.listdir(image_dir):
        ext = os.path.splitext(fn)[1].lower()
        if ext not in IMAGE_EXTS:
            continue
        img_path = os.path.join(image_dir, fn)
        img = imread(img_path)
        if len(img.shape) != 3 or img.shape[2] != 3:
            print('Skipping one-channel image:', img_path)
            continue
        if image_size is not None:
            img = imresize(img, (image_size, image_size))
        images.append(np.array(img))
    print('Found %d images' % len(images))
    return images

# Load images from a list of directories
def load_images_from_list(file_list, image_size):
    images = []
    with open(file_list, 'r') as f:
        for line in f:
            image_dir = line.strip()
            images.extend(load_images(image_dir, image_size))
    return images

# Load images from a super directory
def load_images_from_superdir(superdir, image_size):
    images = []
    for subdir in os.listdir(superdir):
        subdir_path = os.path.join(superdir, subdir)
        if os.path.isdir(subdir_path):
            images.extend(load_images(subdir_path, image_size))
    return images

# Compute Inception Score
def get_inception_score(args, images):
    splits = args.num_splits
    layout = args.tensor_layout

    assert isinstance(images, list)
    assert isinstance(images[0], np.ndarray)
    assert len(images[0].shape) == 3

    inps = [np.expand_dims(img.astype(np.float32), 0) for img in images]
    bs = 1

    with tf.compat.v1.Session() as sess:
        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))
        for i in trange(n_batches, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            sys.stdout.write(".")
            sys.stdout.flush()
            inp = np.concatenate(inps[(i * bs):min((i + 1) * bs, len(inps))], 0)
            if layout == 'NCHW':
                inp = inp.transpose(0, 2, 3, 1)
            pred = sess.run(softmax, {'ExpandDims:0': inp})
            preds.append(pred)
        preds = np.concatenate(preds, 0)

        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)

# Initialize Inception model
def _init_inception():
    global softmax
    MODEL_DIR = './tmp/imagenet'
    DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    
    with tf.io.gfile.GFile(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    
    with tf.compat.v1.Session() as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op in ops:
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s if s is not None else None for s in shape]
                o.set_shape(tf.TensorShape(shape))
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
        softmax = tf.nn.softmax(logits)

# Main function to handle script execution
def main(args):
    got_npy_file = args.input_npy_file is not None
    got_image_dir = args.input_image_dir is not None
    got_image_dir_list = args.input_image_dir_list is not None
    got_image_superdir = args.input_image_superdir is not None
    inputs = [got_npy_file, got_image_dir, got_image_dir_list, got_image_superdir]
    if sum(inputs) != 1:
        raise ValueError('Must give exactly one input type')

    if args.input_npy_file is not None:
        images = np.load(args.input_npy_file)
        images = np.split(images, images.shape[0], axis=0)
        images = [img[0] for img in images]
        mean, std = get_inception_score(args, images)
        print('Inception mean: ', mean)
        print('Inception std: ', std)
    elif args.input_image_dir is not None:
        images = load_images(args.input_image_dir, args.image_size)
        mean, std = get_inception_score(args, images)
        print('Inception mean: ', mean)
        print('Inception std: ', std)
    elif got_image_dir_list:
        images = load_images_from_list(args.input_image_dir_list, args.image_size)
        mean, std = get_inception_score(args, images)
        print('Inception mean: ', mean)
        print('Inception std: ', std)
    elif got_image_superdir:
        images = load_images_from_superdir(args.input_image_superdir, args.image_size)
        mean, std = get_inception_score(args, images)
        print('Inception mean: ', mean)
        print('Inception std: ', std)

if __name__ == '__main__':
    if 'softmax' not in globals():
        _init_inception()
    
    args = parser.parse_args()
    main(args)
    print("Test on {} Done".format(args.input_image_dir if args.input_image_dir else args.input_image_superdir))
