import argparse
import uuid
import sys

import torch
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger

from terratorch.cli_tools import LightningInferenceModel
from distillers import SemanticSegmentationDistiller
from students import (
    DeepLabV3_MobileNet_V3_Large,
    DeepLabV3_ResNet50,
    DeepLabV3_ResNet101,
    LRASPP_MobileNet_V3_Large,
)

STUDENT_MODELS = {
    "deeplabv3-mobilenet-v3-large": DeepLabV3_MobileNet_V3_Large,
    "deeplabv3-resnet50": DeepLabV3_ResNet50,
    "deeplabv3-resnet101": DeepLabV3_ResNet101,
    "lraspp-mobilenet-v3-large": LRASPP_MobileNet_V3_Large,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prithvi distillation")

    parser.add_argument(
        "--teacher-config",
        required=True,
        help="Path to teacher config YAML",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        required=True,
        help="Path to teacher model checkpoint",
    )
    parser.add_argument(
        "--student-model",
        required=True,
        choices=STUDENT_MODELS.keys(),
        help="Student model type",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        required=True,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--experiment-name",
        default="experiment",
        help="MLFlow experiment name",
    )
    parser.add_argument(
        "--run-name",
        default=f"{uuid.uuid4().hex[:8]}",
        help="MLFlow run name (default is random)",
    )
    parser.add_argument(
        "--kd-temperature",
        type=float,
        default=2.0,
        help="Knowledge distillation temperature",
    )
    parser.add_argument(
        "--kd-weight",
        type=float,
        default=0.75,
        help="Knowledge distillation loss weight",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    torch.set_float32_matmul_precision("medium")

    sys.argv = [sys.argv[0]]
    config = LightningInferenceModel.from_config(
        args.teacher_config, args.teacher_checkpoint
    )
    teacher = config.model
    datamodule = config.datamodule
    datamodule.batch_size = args.batch_size

    student = STUDENT_MODELS[args.student_model](
        num_channels=len(teacher.hparams.model_args["backbone_bands"]),
        num_classes=teacher.hparams.model_args["num_classes"],
    )

    distiller = SemanticSegmentationDistiller(
        teacher=teacher,
        student=student,
        kd_temperature=args.kd_temperature,
        kd_weight=args.kd_weight,
    )

    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )

    trainer = Trainer(
        max_epochs=args.num_epochs, logger=mlf_logger, log_every_n_steps=1
    )
    trainer.fit(distiller, datamodule)
    trainer.test(distiller, datamodule)


if __name__ == "__main__":
    main()
