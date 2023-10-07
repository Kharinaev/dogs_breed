import torchvision.models as models
from torch import nn


class ResNetClassificator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.head = nn.Linear(1000, n_classes)

    def forward(self, x):
        return self.head(self.resnet(x))
