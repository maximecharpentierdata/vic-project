import os

import cv2
from tqdm import tqdm


def load_and_preprocess(image_path, segmentation_method="otsu"):
    image = cv2.imread(image_path.__str__())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = make_segmentation(image, segmentation_method)
    return image


def make_segmentation(image, method="otsu"):
    if method == "otsu":
        _, segmented_image = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)
    elif method == "identity":
        segmented_image = image
    else:
        raise NotImplementedError(f"{method} not implemented yet")
    return segmented_image


def get_descriptors(image, descriptors_proportion=0.8):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    values = [keypoint.response for keypoint in keypoints]
    order = sorted(range(len(values)), key=lambda i: values[i])
    descriptors = descriptors[order, :]
    return descriptors[: int(len(descriptors) * descriptors_proportion), :]


def run_preprocessing(images_path, descriptors_proportion):
    descriptors_list = []
    for image_path in tqdm(os.listdir(images_path)):
        image = load_and_preprocess(images_path / image_path)
        descriptors = get_descriptors(image, descriptors_proportion)
        descriptors_list.append(descriptors)
    return descriptors_list
