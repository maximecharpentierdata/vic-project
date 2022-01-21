import multiprocessing

from sklearn.cluster import MiniBatchKMeans
import numpy as np
from tqdm import tqdm
import pandas as pd


def make_clustering(descriptors_dict, n_clusters, method="kmeans"):
    descriptors_input = []
    for descriptors in descriptors_dict:
        descriptors_input += list(descriptors_dict[descriptors])
    descriptors_input = np.array(descriptors_input)
    print(f"Descriptors shape: {descriptors_input.shape}")
    if method == "kmeans":
        clustering_model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=multiprocessing.cpu_count() * 1024,
            verbose=1,
        )
        clustering_model.fit(descriptors_input)
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


def extract_features(descriptors_dict, clustering_model):
    features = dict()
    for descriptors in tqdm(descriptors_dict):
        features[descriptors] = get_features(
            descriptors_dict[descriptors], clustering_model
        )
    return features


def make_final_df(features, labels):
    features_df = pd.DataFrame(features).T.reset_index()
    features_df["path"] = features_df["index"]
    features_df = features_df.drop("index", axis=1)
    final_df = pd.merge(features_df, labels, on="path")
    return final_df
