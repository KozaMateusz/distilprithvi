# Distil Prithvi

Knowledge distillation of geospatial foundation models.

## Datasets & Teachers
Datasets and teachers can be obtained on [huggingface](https://huggingface.co/collections/KozaMateusz/distil-prithvi-680ca48149d5d8a9ad3d25e3).

## TODO
* Fix: The ``compute`` method of metric MulticlassJaccardIndex was called before the ``update``
* check the floods dataset
* inference the teacher model once on the whole dataset, obtain logits and cache them for distillation