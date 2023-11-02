import os
import cv2
import torch
import numpy as np
from absl import app
from absl import flags
from PIL import Image as PILImg
import matplotlib.pyplot as plt
from utils.obj_detection import GroundingDINO
from groundingdino.util.inference import annotate
from utils.helpers import (P, show_mask, load, get_clip)
from torchvision.ops import box_convert


# Define flags
flags.DEFINE_string('dataset_path', '/home/user/Datasets/ptg-november23demo/Object_Detection', 'Path to the dataset folder')
flags.DEFINE_float('box_threshold', 0.5, 'Box threshold for GroundingDINO')
flags.DEFINE_float('text_threshold', 0.5, 'Text threshold for GroundingDINO')
flags.DEFINE_float('alpha', 0.5, 'alpha parameter for Proto-CLIP')
flags.DEFINE_float('beta', 0.15, 'beta paramater for Proto-CLIP')
flags.DEFINE_float('pct', 0.2, 'Threshold for Proto-CLIP')
FLAGS = flags.FLAGS


def process_bbox(boxes, h, w):
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(np.int32)
    return xyxy

def main(argv):
    del argv
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_img_proto = torch.load(os.path.join(os.getcwd(), 'pretrained_checkpoints', 'proto-clip', 'image_features.pt')).to(device) # (NxD)
    z_txt_proto = torch.load(os.path.join(os.getcwd(), 'pretrained_checkpoints', 'proto-clip', 'text_features.pt')).to(device) # (NxD)
    class_names = sorted(load(os.path.join(os.getcwd(), 'pretrained_checkpoints', 'proto-clip', 'class_folder_mapping.pkl'), 'class_names').keys())

    frames_dir = os.path.join(FLAGS.dataset_path, "real-world", 'frames')
    # os.makedirs(frames_dir, exist_ok=True)

    # Get a list of image and mask files
    image_files = sorted([f for f in os.listdir(frames_dir)])

    gdino = GroundingDINO(
        './utils/submodules/GroundingDINO',
        './pretrained_checkpoints/gdino/groundingdino_swint_ogc.pth',
        FLAGS.box_threshold,
        FLAGS.text_threshold)

    device, model, preprocess = get_clip()

    out_dir_path = os.path.join(FLAGS.dataset_path, 'annotated_gdino_mobile_sam_pc')

    for img_file in image_files:
        img_file_path = os.path.join(frames_dir, img_file)
        img = PILImg.open(img_file_path)
        image = np.array(img)
        bboxes, logits, class_labels = gdino.detect(image, out_dir_path)
        H, W = image.shape[:2]
        bounding_boxes = process_bbox(bboxes, H, W)
        zq_imgs_flat = []

        with torch.no_grad():
            for box in bounding_boxes:
                x_min, y_min, x_max, y_max = box
                # Crop the bounding box from the image
                cropped_img = image[y_min:y_max, x_min:x_max]
                crop_img_tensor = preprocess(PILImg.fromarray(cropped_img)).unsqueeze(0).to(device)
                image_features = model.encode_image(crop_img_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                zq_imgs_flat.append(image_features)
                # Display the image using Matplotlib
                # plt.imshow(cropped_img)
                # plt.axis('off')  # Turn off axis labels
                # plt.show()
        zq = torch.cat(zq_imgs_flat, dim=0)
        p = P(zq, z_img_proto, z_txt_proto, alpha=FLAGS.alpha, beta=FLAGS.beta)
        max_confs, max_idx = torch.max(p, dim=1)
        cls = [class_names[idx] for idx in max_idx]

        filtered_idxs = [ idx for idx, conf in enumerate(max_confs) if conf > FLAGS.pct]
        if len(filtered_idxs) > 0:
            _bboxes = bboxes[filtered_idxs]
            _max_confs = [ max_confs[idx] for idx in filtered_idxs ]
            _cls = [ cls[idx] for idx in filtered_idxs ]
            annotated_frame = annotate(image_source=cv2.cvtColor(image, cv2.COLOR_RGB2BGR), boxes=_bboxes, logits=_max_confs, phrases=_cls)
            input_boxes, masks = gdino.get_sam_mask(image, _bboxes)

            plt.figure(figsize=(10, 10))
            plt.imshow(annotated_frame)
            # plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            plt.axis('off')
            plt.show()
            
            # plt.savefig(f"{out_dir_path}/annotated_{gdino.box_threshold}_{gdino.text_threshold}_{time.time()}.png")

            # plt.cla()
            # break


if __name__ == '__main__':
    app.run(main)
