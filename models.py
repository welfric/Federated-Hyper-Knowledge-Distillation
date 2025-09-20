from PIL import Image
from os.path import join
import imageio
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models


class EncoderFemnist(nn.Module):
    def __init__(self, code_length):
        super(EncoderFemnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320), code_length)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        z = F.relu(self.fc1(x))
        return z


class CNNFemnist(nn.Module):
    def __init__(self, args, code_length=50, num_classes=62):
        super(CNNFemnist, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = EncoderFemnist(self.code_length)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.code_length, self.num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z, p


class ResNet18(nn.Module):
    def __init__(self, args, code_length=64, num_classes=10):
        super(ResNet18, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = models.resnet18(num_classes=self.code_length)
        self.classifier = nn.Sequential(nn.Linear(self.code_length, self.num_classes))

    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z, p


class ShuffLeNet(nn.Module):
    def __init__(self, args, code_length=64, num_classes=10):
        super(ShuffLeNet, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = models.shufflenet_v2_x1_0(num_classes=self.code_length)
        self.classifier = nn.Sequential(nn.Linear(self.code_length, self.num_classes))

    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z, p


class TempNet(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=128, tau_min=0.1, tau_max=2.0):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.tau_min = tau_min
        self.tau_max = tau_max

    def forward(self, x):
        h = F.relu(self.fc1(x))
        raw_tau = self.fc2(h)  # unbounded scalar
        tau = torch.sigmoid(raw_tau)  # (0,1)
        tau = tau * (self.tau_max - self.tau_min) + self.tau_min
        return tau.mean()


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):

        self.num_classes = num_classes

        super(SimpleCNN, self).__init__()

        # First block
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Second block
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Third block
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        # Fourth block
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        # Fully connected feature layers
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)

        # Final classifier head
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Full forward = feature extractor + classifier
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z, p

    def feature_extractor(self, x):
        # Conv blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)

        x = F.relu(self.bn7(self.conv7(x)))
        x = self.adaptive_pool(x)

        # Flatten + FC feature layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)

        return x  # final 512-D feature vector

    def classifier(self, features):
        return self.fc3(features)
