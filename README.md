# Distil Prithvi

Knowledge distillation of geospatial foundation models.

## Datasets & Teachers
Datasets and teachers can be obtained on [huggingface](https://huggingface.co/collections/KozaMateusz/distil-prithvi-680ca48149d5d8a9ad3d25e3).

## Example Usage
```console
python distilprithvi.py \
  --teacher-config teachers/hls_burn_scars_teacher/burn_scars_config.yaml \
  --teacher-checkpoint teachers/hls_burn_scars_teacher/Prithvi_EO_V2_300M_BurnScars.pt \
  --student-model lraspp \
  --batch-size 16 \
  --num-epochs 50 \
  --experiment-name hls_burn_scars_distillation \
  --run-name hls_burn_scars_distillation_run \
  --kd-temperature 2.0 \
  --kd-weight 0.75
```

## TODO
* Fix: The ``compute`` method of metric MulticlassJaccardIndex was called before the ``update``
* check the floods dataset
* test distillation with 600M prithvi models (trained by us??)
* add learning rate scheduler