import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
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
        max_epochs: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["teacher", "student"])

        self.teacher = teacher
        self.student = student
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        self.lr = lr
        self.kd_stop_epoch = kd_stop_epoch
        self.num_classes = num_classes
        self.max_epochs = max_epochs

        # Precompute constants used every training step
        self._kd_temperature_sq = kd_temperature**2
        self._ce_weight = 1.0 - kd_weight
        # Cache for extra batch keys (computed once on first batch)
        self._batch_extra_keys: Optional[set] = None

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

    def _unpack_batch(self, batch):
        """Unpack image, mask, and any extra keys from a batch dict."""
        x = batch["image"]
        y = batch["mask"].squeeze(1)
        if self._batch_extra_keys is None:
            self._batch_extra_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in self._batch_extra_keys}
        return x, y, rest

    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass through the model."""
        if self.student is None:
            return self.teacher(x, **kwargs).output
        else:
            return self.student(x)["out"]

    def training_step(self, batch):
        """Training step for the distillation process."""
        x, y, rest = self._unpack_batch(batch)

        y_hat_s = self(x, **rest)
        loss_target = self.criterion(y_hat_s, y)

        use_kd = (
            self.kd_weight > 0
            and self.teacher is not None
            and (self.kd_stop_epoch is None or self.current_epoch < self.kd_stop_epoch)
        )
        if use_kd:
            with torch.no_grad():
                y_hat_t = self.teacher(x, **rest).output
            student_log_probs = torch.log_softmax(
                y_hat_s.reshape(-1, self.num_classes) / self.kd_temperature, dim=1
            )
            teacher_probs = torch.softmax(
                y_hat_t.reshape(-1, self.num_classes) / self.kd_temperature, dim=1
            )
            loss_kd = self.kd_criterion(
                student_log_probs,
                teacher_probs,
            ) * self._kd_temperature_sq
            self.log(
                "train/loss_kd",
                loss_kd,
                on_epoch=True,
                on_step=False,
                batch_size=x.shape[0],
            )
            loss = self.kd_weight * loss_kd + self._ce_weight * loss_target
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
        x, y, rest = self._unpack_batch(batch)
        y_hat_s = self(x, **rest)
        loss = self.criterion(y_hat_s, y)
        self.val_metrics.update(y_hat_s.argmax(dim=1), y)
        self.log("val/loss", loss, on_epoch=True, on_step=False, batch_size=x.shape[0])

    def test_step(self, batch):
        """Test step for the distillation process."""
        x, y, rest = self._unpack_batch(batch)
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
        t_max = self.max_epochs if self.max_epochs is not None else self.trainer.max_epochs
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=self.lr * 1e-2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
