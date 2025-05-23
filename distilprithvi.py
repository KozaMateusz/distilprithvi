import argparse
import uuid
import sys

import logging
import pandas as pd
import coloredlogs
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

coloredlogs.install()


STUDENT_MODELS = {
    "lraspp-mobilenet-v3-large": LRASPP_MobileNet_V3_Large,
    "deeplabv3-mobilenet-v3-large": DeepLabV3_MobileNet_V3_Large,
    "deeplabv3-resnet50": DeepLabV3_ResNet50,
    "deeplabv3-resnet101": DeepLabV3_ResNet101,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prithvi distillation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # fmt: off
    train_parser = subparsers.add_parser("train-student", help="Train the student model")
    train_parser.add_argument("--teacher-config", required=True, help="Path to teacher config YAML")
    train_parser.add_argument("--teacher-checkpoint", required=True, help="Path to teacher model checkpoint")
    train_parser.add_argument("--student-model", required=True, choices=STUDENT_MODELS.keys(), help="Student model type")
    train_parser.add_argument("--batch-size", type=int, required=True, help="Batch size for training")
    train_parser.add_argument("--num-epochs", type=int, required=True, help="Number of training epochs")
    train_parser.add_argument("--experiment-name", default="experiment", help="MLFlow experiment name")
    train_parser.add_argument("--run-name", default=f"{uuid.uuid4().hex[:8]}", help="MLFlow run name (default is random)")
    train_parser.add_argument("--kd-temperature", type=float, default=2.0, help="Knowledge distillation temperature")
    train_parser.add_argument("--kd-weight", type=float, default=0.75, help="Knowledge distillation loss weight")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    train_parser.add_argument("--metrics-output", help="Path to save test metrics as CSV")

    test_parser = subparsers.add_parser("test-teacher", help="Test the teacher model")
    test_parser.add_argument("--teacher-config", required=True, help="Path to teacher config YAML")
    test_parser.add_argument("--teacher-checkpoint", required=True, help="Path to teacher model checkpoint")
    test_parser.add_argument("--batch-size", type=int, default=1, help="Batch size for testing")
    test_parser.add_argument("--metrics-output", help="Path to save test metrics as CSV")
    # fmt: on

    return parser.parse_args()


def save_metrics(name, metrics, output_path):
    """Save metrics to a CSV file."""
    df = pd.DataFrame(metrics)
    df.insert(0, "name", name)
    df.to_csv(
        output_path,
        mode="a",
        header=not pd.io.common.file_exists(output_path),
        index=False,
    )
    logging.info("Metrics saved to %s", output_path)


def run_train_student(args):
    """Train the student model."""
    logging.info("Initializing the teacher model...")
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
        ignore_index=teacher.hparams.ignore_index,
        num_classes=teacher.hparams.model_args["num_classes"],
        class_names=teacher.hparams.class_names,
        kd_temperature=args.kd_temperature,
        kd_weight=args.kd_weight,
        lr=args.lr,
    )

    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )

    trainer = Trainer(
        max_epochs=args.num_epochs,
        logger=mlf_logger,
        log_every_n_steps=1,
    )

    logging.info("Starting the training process...")
    trainer.fit(distiller, datamodule)

    logging.info("Testing the model...")
    metrics = trainer.test(distiller, datamodule)

    if args.metrics_output:
        save_metrics(args.run_name, metrics, args.metrics_output)


def run_test_teacher(args):
    """Test the teacher model."""
    logging.info("Initializing the teacher model...")
    config = LightningInferenceModel.from_config(
        args.teacher_config, args.teacher_checkpoint
    )
    teacher = config.model
    datamodule = config.datamodule
    datamodule.batch_size = args.batch_size

    distiller = SemanticSegmentationDistiller(
        teacher=teacher,
        ignore_index=teacher.hparams.ignore_index,
        num_classes=teacher.hparams.model_args["num_classes"],
        class_names=teacher.hparams.class_names,
    )

    trainer = Trainer(logger=False)

    logging.info("Testing the teacher model...")
    metrics = trainer.test(distiller, datamodule)

    if args.metrics_output:
        save_metrics(args.teacher_checkpoint, metrics, args.metrics_output)


def main():
    """Main function to run the script."""
    args = parse_args()

    sys.argv = [sys.argv[0]]
    torch.set_float32_matmul_precision("medium")

    logging.info("Running in %s mode", args.command)
    if args.command == "train-student":
        run_train_student(args)
    elif args.command == "test-teacher":
        run_test_teacher(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
