# Distil Prithvi

Knowledge distillation of geospatial foundation models.

## Datasets & Teachers
Datasets and teachers can be obtained from [Hugging Face](https://huggingface.co/collections/KozaMateusz/distil-prithvi-680ca48149d5d8a9ad3d25e3).

## Example Usage
Train student example:
```console
python distilprithvi.py train-student \
  --teacher-config teachers/sen1floods11_teacher_300M/config.yaml \
  --teacher-checkpoint teachers/sen1floods11_teacher_300M/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt \
  --student-model deeplabv3-resnet101 \
  --batch-size 16 \
  --num-epochs 20 \
  --experiment-name prithvi-distill \
  --run-name trial-resnet101 \
  --kd-temperature 2.0 \
  --kd-weight 0.75 \
  --lr 1e-4 \
  --metrics-output output.csv
```

Test teacher example:
```console
python distilprithvi.py test-teacher \
  --teacher-config teachers/sen1floods11_teacher_300M/config.yaml \
  --teacher-checkpoint teachers/sen1floods11_teacher_300M/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt \
  --batch-size 8 \
  --metrics-output output.csv

```

## TODO
* test distillation with 600M prithvi models (trained by us??)