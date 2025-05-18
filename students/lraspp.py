from torch import nn
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
)


class LRASPP_MobileNet_V3_Large(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()

        self.model = lraspp_mobilenet_v3_large(
            weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT,
        )
        self.model.backbone["0"][0] = nn.Conv2d(
            num_channels,
            16,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )

        self.model.classifier.low_classifier = nn.Conv2d(
            40, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )

        self.model.classifier.high_classifier = nn.Conv2d(
            128, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )

    def forward(self, x):
        return self.model(x)
