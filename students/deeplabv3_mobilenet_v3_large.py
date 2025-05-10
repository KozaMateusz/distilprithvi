from torch import nn
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)


class DeepLabV3MobileNetV3Large(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()

        self.model = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )

        self.model.backbone["0"][0] = nn.Conv2d(
            num_channels,
            16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        self.model.classifier[4] = nn.Conv2d(
            256,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        return self.model(x)
