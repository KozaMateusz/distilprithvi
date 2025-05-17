# Distil Prithvi

Knowledge distillation of geospatial foundation models.

## Datasets & Teachers
Datasets and teachers can be obtained on [huggingface](https://huggingface.co/collections/KozaMateusz/distil-prithvi-680ca48149d5d8a9ad3d25e3).

## Example Usage
hls_burn_scars
```console
python distilprithvi.py \
  --teacher-config teachers/hls_burn_scars_teacher_300M/config.yaml \
  --teacher-checkpoint teachers/hls_burn_scars_teacher_300M/Prithvi_EO_V2_300M_BurnScars.pt \
  --student-model lraspp \
  --batch-size 16 \
  --num-epochs 50 \
  --experiment-name hls_burn_scars_distillation \
  --run-name hls_burn_scars_distillation_run \
  --kd-temperature 2.0 \
  --kd-weight 0.75
```

sen1floods11
```console
python distilprithvi.py \
    --teacher-config teachers/sen1floods11_teacher_300M/config.yaml \
    --teacher-checkpoint teachers/sen1floods11_teacher_300M/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt \
    --student-model lraspp \
    --batch-size 16 \
    --num-epochs 50 \
    --experiment-name sen1floods11_distillation \
    --run-name sen1floods11_run \
    --kd-temperature 2.0 \
    --kd-weight 0.75
```

## TODO
* Fix: Why is the performance worse when using distillation on sen1floods11??
* Fix: The ``compute`` method of metric MulticlassJaccardIndex was called before the ``update`` (Seems to be a torchmetrics bug)
* test distillation with 600M prithvi models (trained by us??)