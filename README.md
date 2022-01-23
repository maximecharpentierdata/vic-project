# VIC-Image classification

Computer vision academic project

## Data fetching

### Training data

Run:

```
python src/data_fetching/fetch_data_kaggle.py
```

This download the dog vs cat Kaggle Dataset.

### PASCAL dataset

Run:

```
python src/data_fetching/fetch_data_pascal.py
```

This will download the PASCAL VOC 2007 data, and prepare the directories as needed for the image classification pipeline.

## Run classification

We explore object detection through the simpler problem of image classification over images of cats and dogs.

You can run the classification script at `src/classification/classification.py` as followed:

```
usage: classification.py [-h] [--segmentation SEGMENTATION] [--descriptors_proportion DESCRIPTORS_PROPORTION]
                         [--clustering_method CLUSTERING_METHOD] [--classification_model CLASSIFICATION_MODEL] [--n_data N_DATA]
                         [--no_clustering | --no-no_clustering] [--use_df | --no-use_df] [--binary | --no-binary]
```

## Check results

Experiments are saved at `experiments/`. You can run result exploration on both dataset with `src/results.py` and saving it in a `.txt` file:

```
python -m src.results > results.txt
```

