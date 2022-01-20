import os
from joblib import dump, load

import pandas as pd

from src.data_preparation.preprocessing import run_preprocessing
from src.data_preparation.features import (
    make_clustering,
    extract_features,
    make_final_df,
)


def run_pipeline(params):
    if params["use_df"]:
        clustering_model = load(
            params["models_path"]
            / f"clustering_{params['clustering_method']}_{params['n_clusters']}_{params['segmentation']}.joblib"
        )
        final_df = pd.read_csv(
            params["final_data_path"]
            / f"df_{params['clustering_method']}_{params['n_clusters']}_{params['segmentation']}.csv",
        )
        return final_df, clustering_model
    else:
        print("Making segmentation & computing descriptors...")
        descriptors_list = run_preprocessing(
            params["images_path"],
            params["descriptors_proportion"],
            params["segmentation"],
            params["n_data"],
        )

        print("Making clustering...")
        if params["do_clustering"]:
            clustering_model = make_clustering(
                descriptors_list, params["n_clusters"], params["clustering_method"]
            )
            if not params["models_path"].exists():
                os.makedirs(params["models_path"])
            dump(
                clustering_model,
                params["models_path"]
                / f"clustering_{params['clustering_method']}_{params['n_clusters']}_{params['segmentation']}.joblib",
            )
        else:
            clustering_model = load(
                params["models_path"]
                / f"clustering_{params['clustering_method']}_{params['n_clusters']}_{params['segmentation']}.joblib"
            )

        print("Extracting features...")
        features = extract_features(descriptors_list, clustering_model)

        labels = pd.read_csv(params["labels_path"])
        final_df = make_final_df(features, labels)
        if not params["final_data_path"].exists():
            os.makedirs(params["final_data_path"])
        final_df.to_csv(
            params["final_data_path"]
            / f"df_{params['clustering_method']}_{params['n_clusters']}_{params['segmentation']}.csv",
            index=False,
        )

        return final_df, clustering_model
