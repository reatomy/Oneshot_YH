import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SiameseConvNet(nn.Module):
    def __init__(self):
        super(SiameseConvNet, self).__init__()

        self.siamese = nn.Sequential(
            nn.Conv2d(1, 64, 10),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 4),
            nn.ReLU()
        )
        self.siamese.apply(self.init_weights)

        self.fc1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.Sigmoid()
        )
        self.fc1.apply(self.init_weights)

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )
        self.fc2.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.normal_(m.bias, 0.5, 0.01)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, 0.2)
            nn.init.normal_(m.bias, 0.5, 0.01)

    def forward(self, img1, img2):
        rep1 = self.siamese(img1)
        rep1 = rep1.reshape(rep1.shape[0], -1)
        rep1 = self.fc1(rep1)

        rep2 = self.siamese(img2)
        rep2 = rep2.reshape(rep2.shape[0], -1)
        rep2 = self.fc1(rep2)

        siam_dist = torch.abs(rep1 - rep2)
        out = self.fc2(siam_dist)
        
        return out
