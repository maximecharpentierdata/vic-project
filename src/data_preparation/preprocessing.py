import os
import pathlib

import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing


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
    elif method == "MSER":
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(image)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        for contour in hulls:
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        segmented_image = cv2.bitwise_and(image, image, mask=mask)
    else:
        raise NotImplementedError(f"{method} not implemented yet")
    return segmented_image


def get_descriptors(image, descriptors_proportion=0.8):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is not None:
        values = [keypoint.response for keypoint in keypoints]
        order = sorted(range(len(values)), key=lambda i: values[i])
        descriptors = descriptors[order, :]
        return descriptors[: int(len(descriptors) * descriptors_proportion), :]


class PreProcessor:
    def __init__(self, images_path, descriptors_proportion, segmentation_method):
        self.images_path = images_path
        self.descriptors_proportion = descriptors_proportion
        self.segmentation_method = segmentation_method

    def __call__(self, image_path):
        image = load_and_preprocess(
            self.images_path / image_path, self.segmentation_method
        )
        descriptors = get_descriptors(image, self.descriptors_proportion)
        if descriptors is not None and descriptors.shape[0] > 0:
            return image_path.__str__(), descriptors


def run_preprocessing(images_path, descriptors_proportion, segmentation_method, n_data):
    preprocessor = PreProcessor(
        images_path, descriptors_proportion, segmentation_method
    )
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        outputs = list(
            tqdm(
                pool.imap(
                    preprocessor,
                    [pathlib.Path(path) for path in os.listdir(images_path)[:n_data]],
                ),
                total=n_data,
            )
        )
    outputs = [output for output in outputs if output is not None]
    descriptors_dict = dict(outputs)
    return descriptors_dict
