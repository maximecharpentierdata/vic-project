import os

from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import pandas as pd


def make_clustering(descriptors_list, n_clusters, method="kmeans"):
    descriptors_input = []
    for descriptors in descriptors_list:
        descriptors_input += list(descriptors)
    if method == "kmeans":
        clustering_model = KMeans(n_clusters=n_clusters)
        clustering_model.fit(descriptors)
    else:
        raise NotImplementedError(f"{method} not implemented yet")
    return clustering_model


def get_features(descriptors, clustering_model):
    predictions = clustering_model.predict(descriptors)
    hist, _ = np.histogram(
        predictions,
        range=(0, clustering_model.n_clusters),
        bins=clustering_model.n_clusters,
    )
    return hist


def extract_features(descriptors_list, clustering_model):
    features = []
    for descriptors in tqdm(descriptors_list):
        features.append(get_features(descriptors, clustering_model))
    return features


def make_final_df(images_path, features, labels):
    path_df = pd.DataFrame(dict(path=os.listdir(images_path)))
    features_df = pd.DataFrame(features)
    final_df = pd.concat([path_df, features_df], axis=1)
    final_df = pd.merge(final_df, labels, on="path")
    return final_df
