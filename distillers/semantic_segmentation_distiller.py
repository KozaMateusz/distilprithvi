import torch
from torch import nn
import lightning as L


class SemanticSegmentationDistiller(L.LightningModule):
    def __init__(
        self,
        teacher,
        student,
        kd_weight=0.75,
        kd_temperature=2.0,
    ):
        super().__init__()
        self.teacher = teacher
        self.teacher.eval()

        self.student = student

        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        self.kd_criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x):
        return self.student(x)["out"]

    def training_step(self, batch):
        x = batch["image"]
        y = batch["mask"].squeeze(1)

        y_hat_s = self(x)
        loss_target = self.teacher.criterion(y_hat_s, y)

        if self.kd_weight == 0:
            loss = loss_target
        else:
            other_keys = batch.keys() - {"image", "mask", "filename"}
            rest = {k: batch[k] for k in other_keys}
            with torch.no_grad():
                y_hat_t = self.teacher(x, **rest).output
            loss_kd = self.kd_criterion(
                torch.log_softmax(y_hat_s / self.kd_temperature, dim=1),
                torch.softmax(y_hat_t / self.kd_temperature, dim=1),
            ) * (self.kd_temperature**2)
            self.log(
                "train/loss_target",
                loss_target,
                on_epoch=True,
                on_step=False,
                batch_size=x.shape[0],
            )
            self.log(
                "train/loss_kd",
                loss_kd,
                on_epoch=True,
                on_step=False,
                batch_size=x.shape[0],
            )

            loss = self.kd_weight * loss_kd + (1 - self.kd_weight) * loss_target

        self.log(
            "train/loss", loss, on_epoch=True, on_step=False, batch_size=x.shape[0]
        )
        self.teacher.train_metrics.update(y_hat_s.argmax(dim=1), y)
        return loss

    def validation_step(self, batch):
        x = batch["image"]
        y = batch["mask"].squeeze(1)
        y_hat_s = self(x)
        loss = self.teacher.criterion(y_hat_s, y)
        self.teacher.val_metrics.update(y_hat_s.argmax(dim=1), y)
        self.log("val/loss", loss, on_epoch=True, on_step=False, batch_size=x.shape[0])

    def test_step(self, batch):
        x = batch["image"]
        y = batch["mask"].squeeze(1)
        y_hat_s = self(x)
        loss = self.teacher.criterion(y_hat_s, y)
        self.teacher.test_metrics[0].update(y_hat_s.argmax(dim=1), y)
        self.log("test/loss", loss, on_epoch=True, on_step=False, batch_size=x.shape[0])

    def on_train_epoch_end(self):
        metrics = self.teacher.train_metrics.compute()
        self.log_dict(metrics, on_epoch=True, on_step=False)
        self.teacher.train_metrics.reset()

        optimizer = self.trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", current_lr, on_epoch=True, on_step=False)

    def on_validation_epoch_end(self):
        metrics = self.teacher.val_metrics.compute()
        self.log_dict(metrics, on_epoch=True, on_step=False)
        self.teacher.val_metrics.reset()

    def on_test_epoch_end(self):
        metrics = self.teacher.test_metrics[0].compute()
        self.log_dict(metrics, on_epoch=True, on_step=False)
        self.teacher.test_metrics[0].reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]
