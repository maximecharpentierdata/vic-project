# VIC-Image classification

Computer vision academic project

## Data fetching

Run:

```
python src/data_fetching/fetch_data.py
```

This will download the PASCAL VOC 2007 data, and prepare the directories as needed for the image classification pipeline.

## Run classification

We explore object detection through the simpler problem of image classification over images of cats and dogs.

You can run the classification script at `src/classification/classification.py` as followed:

```
usage: classification.py [-h] [--segmentation SEGMENTATION]
                         [--descriptors_proportion DESCRIPTORS_PROPORTION]
                         [--clustering_method CLUSTERING_METHOD] [--n_clusters N_CLUSTERS]
                         [--classification_model CLASSIFICATION_MODEL]
                         [--do_clustering DO_CLUSTERING]

optional arguments:
  -h, --help            show this help message and exit
  --segmentation SEGMENTATION
  --descriptors_proportion DESCRIPTORS_PROPORTION
  --clustering_method CLUSTERING_METHOD
  --n_clusters N_CLUSTERS
  --classification_model CLASSIFICATION_MODEL
  --do_clustering DO_CLUSTERING
```