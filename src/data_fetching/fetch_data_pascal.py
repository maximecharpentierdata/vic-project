import os
import pathlib
import pandas as pd
import shutil
import urllib.request
import tarfile

import pandas as pd
from tqdm import tqdm

RAW_DATA_PATH = pathlib.Path("./data/raw/")
RAW_IMAGES_PATH = RAW_DATA_PATH / "JPEGImages"
INTERIM_DATA_PATH = pathlib.Path("./data/pascal/")
INTERIM_IMAGES_PATH = INTERIM_DATA_PATH / "images"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def prepare_data():
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
    download_url(url, "./VOCtrainval_06-Nov-2007.tar")
    with tarfile.open("./VOCtrainval_06-Nov-2007.tar") as tar_file:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar_file, "./")

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
    if not INTERIM_IMAGES_PATH.exists():
        os.makedirs(INTERIM_IMAGES_PATH)
    for image in images:
        shutil.copy(RAW_IMAGES_PATH / image, INTERIM_IMAGES_PATH / image)


def prepare_label_dataset(cat_pictures, dog_pictures):
    cat_pictures = list(cat_pictures)
    dog_pictures = list(dog_pictures)

    labels = pd.DataFrame(dict(path=cat_pictures, label=["cat"] * len(cat_pictures)))

    labels = labels.append(
        pd.DataFrame(dict(path=dog_pictures, label=["dog"] * len(dog_pictures)))
    )
    labels = labels.sample(frac=1, random_state=42)
    labels.to_csv(INTERIM_DATA_PATH / "labels.csv", index=False)


if __name__ == "__main__":
    prepare_data()

    cat_pictures = get_cat_pictures()
    dog_pictures = get_dog_pictures()

    cat_pictures, dog_pictures = cat_pictures.difference(
        dog_pictures
    ), dog_pictures.difference(cat_pictures)

    prepare_image_dataset(cat_pictures, dog_pictures)
    prepare_label_dataset(cat_pictures, dog_pictures)
