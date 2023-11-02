import os
import cv2
import numpy as np
import pandas as pd
from absl import logging
import matplotlib.pyplot as plt
import torch
import pickle
import torch.nn.functional as F
from tqdm import tqdm
import clip
from easydict import EasyDict
import yaml


def read_xlsx_and_create_dict(xlsx_file, dataset_path):
    """
    Reads an XLSX file and creates a dictionary mapping class names to sequences.

    Args:
        xlsx_file (str): Path to the input XLSX file.
        dataset_path (str): Path to the dataset folder.

    Returns:
        dict: A dictionary with class names as keys and lists of sequences as values.

    Raises:
        Exception: If an error occurs while reading the XLSX file.
    """
    try:
        df = pd.read_excel(xlsx_file)

        # Process the 'Sequences' column, excluding empty values
        df['Sequences'] = df.apply(lambda row: [os.path.join(dataset_path, ("RecipeObjects" if row['Recipe'] != "Others" else "OtherObjects"), seq) for seq in row['Sequences'].split(', ') if pd.notna(seq) and seq.strip()] if pd.notna(row['Sequences']) else [], axis=1)

        # Group by 'Object' (class name) and concatenate the sequences into lists
        class_folder_mapping = df.groupby('Object')['Sequences'].sum().to_dict()

        # Remove classes with empty sequences
        class_folder_mapping = {k: v for k, v in class_folder_mapping.items() if v}

        return class_folder_mapping
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return {}


def get_cropped_image(image, mask, vis=False):
    """
    Get a cropped image (bounding box) using a mask where pixel value is 1.

    Args:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The mask where pixel value is 1.
        vis (bool): Flag to indicate whether to plot the iamge, mask and cropped image for debugging purposes
    Returns:
        numpy.ndarray: The cropped image if an object is found in the mask, or None if not found.
    """
    try:
        # Create a binary mask where object is 255 (white) and background is 0 (black)
        binary_mask = np.uint8(mask > 0)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # No contours found, no object in the mask
            return None

        # Get the largest contour (assumed to be the object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image using the bounding box
        cropped_image = image[y:y+h, x:x+w]

        if vis:
            plot_images_with_mask_and_cropped(image, mask, cropped_image)

        return cropped_image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def plot_images_with_mask_and_cropped(image, mask, cropped_image):
    """
    Load and display an image, its corresponding mask, and the cropped image side by side.

    Args:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The mask where pixel value is 1.
        cropped_image_path (numpy.ndarray): Path to the cropped image.
    """

    # Create a 1x3 grid for displaying the images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Display the original image
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Display the mask image
    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Mask")
    axs[1].axis("off")

    # Display the cropped image
    axs[2].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Cropped Image")
    axs[2].axis("off")

    # Show the plot
    plt.show()


def save_cropped_image(cropped_image, class_output_dir, image_name):
    """
    Save a cropped image to the specified directory.

    Args:
        cropped_image (numpy.ndarray): The cropped image to be saved.
        class_output_dir (str): The directory where the image should be saved.
        image_name (str): The name of the image file.

    Returns:
        None

    Raises:
        cv2.error: If an error occurs while saving the image.

    """
    if cropped_image is not None and cropped_image.size > 0:
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

        if os.path.isdir(class_output_dir) and image_name:
            image_path = os.path.join(class_output_dir, image_name)

            try:
                if cv2.imwrite(image_path, cropped_image):
                    logging.info(f"Image saved successfully to: {image_path}")
                else:
                    logging.error("Error: Failed to save the image.")
            except cv2.error as e:
                logging.error(f"Error occurred while saving the image: {e}")
        else:
            logging.error("Error: Invalid class_output_dir or image_name.")
    else:
        logging.warning("No object found in the mask.")


def get_seed():
    """
    Returns a random number generator seed for reproducibility purposes
    """
    return 1


def dir_exists(path):
    """
    Returns boolean value depending on whether the given path exists
    """
    return os.path.exists(path)


def beautify(string):
    return string.strip().replace('/', '_').replace('-', '_')


# def get_clip():
#     # Load the model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load('ViT-L/14', device)
#     return (device, model, preprocess)

def get_clip(device):
    # Load the model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-L/14', device)
    return ( model, preprocess)


def get_model_dir_root(cfg):
    return f"{cfg['cache_dir']}/models/{beautify(cfg['backbone'])}/K-{cfg['shots']}"

def save(obj, filepath, msg):
    """
    Saves the input object as a pickle file
    """
    print(f"Saving {msg} to {filepath}")
    with open(filepath, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(filepath, msg):
    """
    Loads a pickle file from disk
    """
    print(f"Loading {msg} from {filepath}")
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)

def get_target_dir():
    return "cropped_objs"

def scale_bboxes_wrt_H_W(bboxes, H, W):
    for i in range(bboxes.size(0)):
        bboxes[i] = bboxes[i] * torch.Tensor([W, H, W, H])
        bboxes[i][:2] -= bboxes[i][2:] / 2
        bboxes[i][2:] += bboxes[i][:2]
    return bboxes

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def cv2_read(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_config(box_threshold, text_threshold):
    config_file = 'config.yml'
    config_absolute_path =  os.path.join(os.getcwd(), config_file)
    
    # Read data from the YAML file
    logging.info(f"Reading config file: {config_absolute_path}")
    with open(config_absolute_path, 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)

    config = EasyDict(**config_dict)

    # for hyperparameter search
    config.BOX_THRESHOLD = box_threshold
    config.TEXT_THRESHOLD = text_threshold

    return config

def P(zq_imgs_flat, z_img_proto, z_text_proto, alpha, beta):
    """
    Returns probability dist, p = alpha * p_i + (1-alpha) * p_t
    """
    # compute pairwise euclidean distances(query, prototypes)
    xq_img_proto_dists = torch.cdist(
        zq_imgs_flat.float(), z_img_proto.float(), p=2).pow(2)
    xq_text_proto_dists = torch.cdist(
        zq_imgs_flat.float(), z_text_proto.float(), p=2).pow(2)

    # P(y=k|query_image,support_images)
    p_i = F.softmax(beta*(-xq_img_proto_dists), dim=1)

    #  P(y=k|query_image,support_text)
    p_t = F.softmax(beta*(-xq_text_proto_dists), dim=1)

    # total probability = alpha * p_image + (1-alpha) - p_text
    p = alpha * p_i + (1-alpha) * p_t

    return p

def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return classnames, clip_weights


def get_textual_memory_bank(cfg, classnames, template, clip_model):
    msg = "text_memory_bank"
    model_dir_root = get_model_dir_root(cfg)
    os.makedirs(model_dir_root, exist_ok=True)
    path = os.path.join(
        model_dir_root, f"text_mb_{beautify(cfg['backbone'])}_K_{cfg['shots']}.pkl")

    if dir_exists(path):
        text_prompts = classnames
        return text_prompts, load(path, msg)
    else:
        # Textual features
        text_prompts, textual_memory_bank = clip_classifier(
            classnames, template, clip_model)
        save(textual_memory_bank, path, msg)
        return text_prompts, textual_memory_bank


def build_cache_model(cfg, clip_model, train_loader_cache):
    model_dir_root = get_model_dir_root(cfg) + '/aug'
    os.makedirs(model_dir_root, exist_ok=True)

    def get_filename(cfg, type):
        return f"{model_dir_root}/visual_mb_{type}_aug_{cfg['augment_epoch']}_{cfg['shots']}_shots.pt"

    key_path = get_filename(cfg, 'keys')
    value_path = get_filename(cfg, 'values')

    if dir_exists(key_path) and dir_exists(value_path):
        cache_keys = torch.load(key_path)
        cache_values = torch.load(value_path)
    else:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print(
                    'Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(
                    torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)

        # sorting
        cache_values = torch.cat(cache_values, dim=0)
        index = torch.argsort(cache_values)
        cache_values = cache_values[index]
        cache_keys = cache_keys[:, index]
        cache_values = F.one_hot(cache_values)

        torch.save(cache_keys, key_path)
        torch.save(cache_values, value_path)

    return cache_keys, cache_values