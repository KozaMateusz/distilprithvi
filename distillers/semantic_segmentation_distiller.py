import torch
from torch import nn
import lightning as L
from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)
from terratorch.tasks.base_task import TerraTorchTask
from typing import Optional


class SemanticSegmentationDistiller(L.LightningModule):
    """
    Distillation module for semantic segmentation tasks.
    This module is designed to transfer knowledge from a teacher modelto a student model.
    """

    def __init__(
        self,
        ignore_index: int,
        num_classes: int,
        class_names: list,
        kd_weight: float = 0.75,
        kd_temperature: float = 2.0,
        lr: float = 1e-4,
        teacher: Optional[TerraTorchTask] = None,
        student: Optional[nn.Module] = None,
        kd_stop_epoch: Optional[int] = None,
    ):
        super().__init__()

        self.teacher = teacher
        self.student = student
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        self.lr = lr
        self.kd_stop_epoch = kd_stop_epoch

        self._validate_args()

        if teacher is not None:
            self.teacher.eval()
            self.teacher.freeze()

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.kd_criterion = nn.KLDivLoss(reduction="batchmean")

        metrics = self._create_metrics(ignore_index, num_classes, class_names)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def _validate_args(self):
        """Validate the arguments provided to the distiller."""
        if self.teacher is None and self.student is None:
            raise ValueError(
                "Both teacher and student models are None. At least one is required."
            )
        if self.teacher is None and self.kd_weight > 0:
            raise ValueError("KD weight > 0 requires a teacher model.")
        if self.kd_weight < 0 or self.kd_weight > 1:
            raise ValueError("KD weight must be between 0 and 1.")
        if self.kd_temperature <= 0:
            raise ValueError("KD temperature must be greater than 0.")
        if self.lr <= 0:
            raise ValueError("Learning rate must be greater than 0.")

    def _create_metrics(self, ignore_index: int, num_classes: int, class_names: list):
        """Create the metrics for training, validation, and testing."""
        metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
                "classwise_accuracy": ClasswiseWrapper(
                    MulticlassAccuracy(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        multidim_average="global",
                        average=None,
                    ),
                    prefix="accuracy_",
                    labels=class_names,
                ),
                "iou_micro": MulticlassJaccardIndex(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                "iou": MulticlassJaccardIndex(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ),
                "classwise_iou": ClasswiseWrapper(
                    MulticlassJaccardIndex(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    prefix="iou_",
                    labels=class_names,
                ),
                "f1_score": MulticlassF1Score(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
            }
        )
        return metrics

    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass through the model."""
        if self.student is None:
            return self.teacher(x, **kwargs).output
        else:
            return self.student(x)["out"]

    def training_step(self, batch):
        """Training step for the distillation process."""
        x = batch["image"]
        y = batch["mask"].squeeze(1)
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}

        y_hat_s = self(x, **rest)
        loss_target = self.criterion(y_hat_s, y)

        use_kd = (
            self.kd_weight > 0
            and self.teacher is not None
            and (self.kd_stop_epoch is None or self.current_epoch < self.kd_stop_epoch)
        )

        if use_kd:
            y_hat_t = self.teacher(x, **rest).output
            loss_kd = self.kd_criterion(
                torch.log_softmax(y_hat_s / self.kd_temperature, dim=1),
                torch.softmax(y_hat_t / self.kd_temperature, dim=1),
            ) * (self.kd_temperature**2)
            self.log(
                "train/loss_kd",
                loss_kd,
                on_epoch=True,
                on_step=False,
                batch_size=x.shape[0],
            )
            loss = self.kd_weight * loss_kd + (1 - self.kd_weight) * loss_target
        else:
            loss = loss_target

        self.log_dict(
            {
                "train/loss_target": loss_target,
                "train/loss": loss,
                "train/use_kd": use_kd,
            },
            on_epoch=True,
            on_step=False,
            batch_size=x.shape[0],
        )
        self.train_metrics.update(y_hat_s.argmax(dim=1), y)
        return loss

    def validation_step(self, batch):
        """Validation step for the distillation process."""
        x = batch["image"]
        y = batch["mask"].squeeze(1)
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}
        y_hat_s = self(x, **rest)
        loss = self.criterion(y_hat_s, y)
        self.val_metrics.update(y_hat_s.argmax(dim=1), y)
        self.log("val/loss", loss, on_epoch=True, on_step=False, batch_size=x.shape[0])

    def test_step(self, batch):
        """Test step for the distillation process."""
        x = batch["image"]
        y = batch["mask"].squeeze(1)
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}
        y_hat_s = self(x, **rest)
        loss = self.criterion(y_hat_s, y)
        self.test_metrics.update(y_hat_s.argmax(dim=1), y)
        self.log("test/loss", loss, on_epoch=True, on_step=False, batch_size=x.shape[0])

    def on_train_epoch_end(self):
        """End of training epoch."""
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_epoch=True, on_step=False)
        self.train_metrics.reset()

        optimizer = self.trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", current_lr, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        """End of validation epoch."""
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_epoch=True, on_step=False)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        """End of test epoch."""
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_epoch=True, on_step=False)
        self.test_metrics.reset()

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]
