from nis import cat
import os
import pathlib
import pandas as pd
import shutil
import urllib.request
from tqdm import tqdm
import tarfile

RAW_DATA_PATH = pathlib.Path("./data/raw/")
RAW_IMAGES_PATH = RAW_DATA_PATH / "JPEGImages"
FINAL_DATA_PATH = pathlib.Path("./data/final/")
FINAL_IMAGES_PATH = FINAL_DATA_PATH / "images"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def prepare_data():
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
    download_url(url, './VOCtrainval_06-Nov-2007.tar')
    with tarfile.open("./VOCtrainval_06-Nov-2007.tar") as tar_file:
        tar_file.extractall("./")

    shutil.move("./VOCdevkit/VOC2007", RAW_DATA_PATH)
    os.remove("./VOCtrainval_06-Nov-2007.tar")
    shutil.rmtree("./VOCdevkit")


def get_cat_pictures():
    cat_pictures = set()
    with open(RAW_DATA_PATH / "ImageSets/Main/cat_trainval.txt") as file:
        for line in file.readlines():
            if line.split()[1] == "1":
                cat_pictures.add(line.split()[0] + ".jpg")
    return cat_pictures


def get_dog_pictures():
    dog_pictures = set()
    with open(RAW_DATA_PATH / "ImageSets/Main/dog_trainval.txt") as file:
        for line in file.readlines():
            if line.split()[1] == "1":
                dog_pictures.add(line.split()[0] + ".jpg")
    return dog_pictures


def prepare_image_dataset(cat_pictures, dog_pictures):
    images = cat_pictures.union(dog_pictures)
    if not FINAL_IMAGES_PATH.exists():
        os.makedirs(FINAL_IMAGES_PATH)
    for image in images:
        shutil.copy(RAW_IMAGES_PATH / image, FINAL_IMAGES_PATH / image)


def prepare_label_dataset(cat_pictures, dog_pictures):
    cat_pictures = list(cat_pictures)
    dog_pictures = list(dog_pictures)

    labels = pd.DataFrame(
        dict(
            path = cat_pictures,
            label = ["cat"] * len(cat_pictures)
        )
    )

    labels = labels.append(
        pd.DataFrame(
            dict(
                path=dog_pictures,
                label = ["dog"] * len(dog_pictures)
            )
        )
    )
    labels = labels.sample(frac=1, random_state=42)
    labels.to_csv(FINAL_DATA_PATH / "labels.csv", index=False)


if __name__ == "__main__":
    prepare_data()

    cat_pictures = get_cat_pictures()
    dog_pictures = get_dog_pictures()

    cat_pictures, dog_pictures = cat_pictures.difference(dog_pictures), dog_pictures.difference(cat_pictures)

    prepare_image_dataset(cat_pictures, dog_pictures)
    prepare_label_dataset(cat_pictures, dog_pictures)
