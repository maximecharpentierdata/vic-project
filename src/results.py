import pathlib
from joblib import load
import json
import os

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import numpy as np

from src.classification.classification import show_params, prepare_data_for_training
from src.data_preparation.preprocessing import run_preprocessing
from src.data_preparation.features import extract_features, make_final_df


def load_model(experiment_path):
    model = load(experiment_path / "model.joblib")
    return model


def load_clustering_model(experiment_path):
    clustering_model = load(experiment_path / "clustering_model.joblib")
    return clustering_model


def load_final_df(experiment_path):
    final_df = pd.read_csv(experiment_path / "final_df.csv")
    return final_df


def load_params(experiment_path):
    with open(experiment_path / "params.json") as file:
        params = json.load(file)
    return params


def show_matrix(y_pred, y):
    confusion_matrix_ = confusion_matrix(y, y_pred, labels=["cat", "dog"])
    ConfusionMatrixDisplay(confusion_matrix_, display_labels=["cat", "dog"]).plot()
    return confusion_matrix_


def show_metrics(confusion_matrix_):
    accuracy = np.trace(confusion_matrix_) / np.sum(confusion_matrix_)
    cat_precision = confusion_matrix_[0, 0] / (
        confusion_matrix_[0, 0] + confusion_matrix_[0, 1]
    )
    dog_precision = confusion_matrix_[1, 1] / (
        confusion_matrix_[1, 1] + confusion_matrix_[1, 0]
    )

    cat_recall = confusion_matrix_[0, 0] / (
        confusion_matrix_[0, 0] + confusion_matrix_[1, 0]
    )
    dog_recall = confusion_matrix_[1, 1] / (
        confusion_matrix_[1, 1] + confusion_matrix_[0, 1]
    )

    average_precision = (cat_precision + dog_precision) / 2

    print(
        f"""\n
        Accuracy: {accuracy*100:.2f} % 

        Cat precision: {cat_precision*100:.2f} %
        Cat recall: {cat_recall*100:.2f} %

        Dog precision: {dog_precision*100:.2f} %
        Dog recall: {dog_recall*100:.2f} %

        Average precision: {average_precision*100:.2f} %
        """
    )


def run_general(final_df, model, params):
    X_scaled, y = prepare_data_for_training(
        final_df.drop("path", axis=1), binary=params["binary"]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=1 / 3, random_state=42
    )
    y_pred = model.predict(X_test)

    confusion_matrix_ = show_matrix(y_pred, y_test)
    show_metrics(confusion_matrix_)


def run_pascal(model, clustering_model, params):
    pascal_path = pathlib.Path("./data/pascal")
    images_path = pascal_path / "images"
    labels_path = pascal_path / "labels.csv"

    descriptors_dict = run_preprocessing(
        images_path, 1, "MSER", len(os.listdir(images_path))
    )
    features = extract_features(descriptors_dict, clustering_model)

    labels = pd.read_csv(labels_path)
    pascal_df = make_final_df(features, labels)

    run_general(pascal_df, model, params)


def main(experiments_path, date):
    experiment_path = experiments_path / date

    model = load_model(experiment_path)
    clustering_model = load_clustering_model(experiment_path)
    final_df = load_final_df(experiment_path)
    params = load_params(experiment_path)

    show_params(params)

    run_general(final_df, model, params)

    print("TEST ON PASCAL:")

    run_pascal(model, clustering_model, params)


if __name__ == "__main__":
    EXPERIMENTS_PATH = pathlib.Path("./experiments")

    for date in os.listdir(EXPERIMENTS_PATH):
        main(EXPERIMENTS_PATH, date)
