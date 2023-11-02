import os
import time
import torch
import numpy as np
from PIL import Image as Img
import matplotlib.pyplot as plt

# import logging
from typing import Tuple
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, load_image, predict, annotate

from utils.helpers import show_box, show_mask, get_config, scale_bboxes_wrt_H_W

# from segment_anything import SamPredictor, sam_model_registry
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# TODO (jishnu): remove unnecessary imports, comments and print statements


class ObjectDetector:
    def __init__(self, box_threshold, text_threshold) -> None:
        self.cfg = get_config(box_threshold, text_threshold) # config contents
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.sam = sam_model_registry["vit_h"](checkpoint="pretrained_checkpoints/sam/sam_vit_h_4b8939.pth")
        self.sam = sam_model_registry["vit_t"](checkpoint="pretrained_checkpoints/sam/mobile_sam.pt")
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def get_sam_mask(self, image, prompt_bboxes):
        H, W = image.shape[:2]
        prompt_bboxes = scale_bboxes_wrt_H_W(prompt_bboxes, H, W)
        input_boxes = torch.tensor(prompt_bboxes, device=self.predictor.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        self.predictor.set_image(image)
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return input_boxes, masks

    def detect(self):
        # override this method in the inherited classes
        raise NotImplementedError


class GroundingDINO(ObjectDetector):
    def __init__(self, dino_root_path: str, pretrained_ckpt_path: str, box_threshold: float, text_threshold: float):
        """
        Initialize the GroundingDINO.

        Args:
            dino_root_path (str): Path to the DINO root directory.
            pretrained_ckpt_path (str): Path to the pretrained checkpoint for DINO model.
            box_threshold (float): box threshold paramter for GroundingDINO model
            text_threshold (float): text threshold paramter for GroundingDINO model
        """
        super().__init__(box_threshold, text_threshold)
        self.dino_root_path = dino_root_path
        self.dino_py = os.path.join(self.dino_root_path, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.model_dino = None  # Will be loaded in self.load_dino_model()
        self.text_prompt = " . ".join(self.cfg.OBJ_DETECTION.OBJECTS)
        self.box_threshold = self.cfg.OBJ_DETECTION.BOX_THRESHOLD
        self.text_threshold = self.cfg.OBJ_DETECTION.TEXT_THRESHOLD
        self.model_name = self.cfg.OBJ_DETECTION.METHOD
        self.arxiv_link = self.cfg.OBJ_DETECTION.ARXIV_LINK
        
        self.load_dino_model()

    def load_dino_model(self):
        """
        Load the DINO model using the specified paths.
        """
        try:
            self.model_dino = load_model(self.dino_py, self.pretrained_ckpt_path)
        except Exception as e:
            raise Exception("Error loading DINO model:", e)


    def load_image(self, image: np.ndarray) -> Tuple[np.array, torch.Tensor]:
        """
        Modification of GroundingDINO/groundingdino/util/inference.py
        to directly operate on image ndarray instead of image_path
        """
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        image_transformed, _ = transform(Img.fromarray(image), None)
        return image, image_transformed


    def detect(self, image: np.array):
        """
        Detect indoor furniture objects in an image and save the annotated result.

        Args:
            image (np.array): Input image as numpy array.
            out_dir_path (str): Path to the output directory.
        """
        # os.makedirs(out_dir_path, exist_ok=True)

        _, transformed_image = self.load_image(image)

        try:

            boxes, logits, class_labels = predict(
                model=self.model_dino,
                image=transformed_image,
                caption=self.text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold
            )

            # annotated_frame = annotate(image_source=image, boxes=boxes, logits=logits, phrases=class_labels)
            # input_boxes, masks = self.get_sam_mask(image, boxes)
            # # annotated_frame = annotate(image_source=image, boxes=input_boxes.cpu(), logits=logits, phrases=class_labels)

            # plt.figure(figsize=(5, 5))
            # plt.imshow(annotated_frame)
            # # plt.imshow(image)
            # for mask in masks:
            #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            # # for box in input_boxes:
            # #     show_box(box.cpu().numpy(), plt.gca())
            # plt.axis('off')
            # # plt.show()
            
            # plt.savefig(f"{out_dir_path}/annotated_{self.box_threshold}_{self.text_threshold}_{time.time()}.png")

            # plt.cla()

            return boxes, logits, class_labels

        except Exception as e:
            raise Exception("Error during detection:", e)
