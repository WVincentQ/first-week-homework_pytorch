import torch.nn as nn
import torch


class Net(nn.Module):

    def __init__(self, num_classes=6):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.bn1 = nn.BatchNorm2d(64)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.bn2 = nn.BatchNorm2d(128)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, stride=1, bias=False)  # unsqueeze channels
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.bn3 = nn.BatchNorm2d(256)


        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(0.4)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.maxpool2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.maxpool3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        x = self.fc1(out)
        x = self.fc2(x)



        return x


def net(num_classes=1000):
    return Net(num_classes=num_classes)
