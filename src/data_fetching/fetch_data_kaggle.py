import pathlib
import os
import shutil

import pandas as pd
from tqdm import tqdm

RAW_DATA_PATH = pathlib.Path("./data/raw/")
RAW_IMAGES_PATH = RAW_DATA_PATH / "JPEGImages"
INTERIM_DATA_PATH = pathlib.Path("./data/interim/")
INTERIM_IMAGES_PATH = INTERIM_DATA_PATH / "images"


def download_data():
    os.system("kaggle competitions download -c dogs-vs-cats")
    os.system("unzip dogs-vs-cats.zip")
    os.system("unzip train.zip")
    os.remove("dogs-vs-cats.zip")
    os.remove("train.zip")
    os.remove("test1.zip")
    os.remove("sampleSubmission.csv")


def process_data(images_path):
    dogs = []
    cats = []
    if not INTERIM_IMAGES_PATH.exists():
        os.makedirs(INTERIM_IMAGES_PATH)
    for image_name in tqdm(os.listdir(images_path)):
        if image_name.split(".")[0] == "dog":
            dogs.append(image_name)
        else:
            cats.append(image_name)
        shutil.copy(images_path / image_name, INTERIM_IMAGES_PATH / image_name)

    labels = (
        pd.DataFrame(dict(label=["cat"] * len(cats), path=cats))
        .append(
            pd.DataFrame(dict(label=["dog"] * len(dogs), path=dogs)), ignore_index=True
        )
        .sample(frac=1)
        .reset_index(drop=True)
    )

    labels.to_csv(INTERIM_DATA_PATH / "labels.csv", index=False)


if __name__ == "__main__":
    download_data()
    process_data(pathlib.Path("./train/"))
    shutil.rmtree("./train")
