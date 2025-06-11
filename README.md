# Distil Prithvi

Knowledge distillation of geospatial foundation models.

## Datasets & Teachers
Datasets and teachers can be obtained from [Hugging Face](https://huggingface.co/collections/KozaMateusz/distil-prithvi-680ca48149d5d8a9ad3d25e3).

## Example Usage
Train student help:
```console
python distilprithvi.py train-student -h
```

Test teacher help:
```console
python distilprithvi.py test-teacher -h
```

## TODO:
* Fix epochs logging in MLFLOW
