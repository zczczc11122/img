import torchvision

import torch.nn as nn


class Resnet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.base_model = getattr(torchvision.models, args.arch)(pretrained=True)
        self.feature_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, args.num_class)
    def forward(self, x):
        feature = self.base_model(x)
        logits = self.classifier(feature)
        return logits