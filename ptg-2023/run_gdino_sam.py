import os
import torch
from absl import app
from absl import flags
from absl import logging
from utils.helpers import get_target_dir
from utils.obj_detection import GroundingDINO
from PIL import Image as PILImg
import numpy as np


# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_path', '/home/user/Datasets/ptg-november23demo/Object_Detection', 'Path to the dataset folder')
flags.DEFINE_float('box_threshold', 0.3, 'Box threshold for GroundingDINO')
flags.DEFINE_float('text_threshold', 0.3, 'Text threshold for GroundingDINO')

def main(argv):
    del argv

    frames_dir = os.path.join(FLAGS.dataset_path, "real-world", 'frames')

    # Get a list of image and mask files
    image_files = sorted([f for f in os.listdir(frames_dir)])

    gdino = GroundingDINO(
        './utils/submodules/GroundingDINO',
        './pretrained_checkpoints/gdino/groundingdino_swint_ogc.pth',
        FLAGS.box_threshold,
        FLAGS.text_threshold)

    out_dir_path = os.path.join(FLAGS.dataset_path, 'annotated_gdino_mobile_sam', f"{FLAGS.box_threshold}-{FLAGS.text_threshold}")

    for img_file in image_files:
        img_file_path = os.path.join(frames_dir, img_file)
        print(img_file_path)
        img = np.array(PILImg.open(img_file_path))
        boxes, masks, logits, class_labels = gdino.detect(img, out_dir_path)


if __name__ == '__main__':
    app.run(main)
