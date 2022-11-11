import torch 
from torch import nn

class Large_Generator(nn.Module):

    def __init__(self, in_ch):
        super(Large_Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 2, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.bn5 = nn.BatchNorm2d(64)
        self.deconv6 = nn.ConvTranspose2d(64, in_ch, 2, stride=2)

    def forward(self, x):
        h = torch.nn.functional.leaky_relu(self.bn1(self.conv1(x)))
        h = torch.nn.functional.leaky_relu(self.bn2(self.conv2(h)))
        h = torch.nn.functional.leaky_relu(self.bn3(self.conv3(h)))
        h = torch.nn.functional.leaky_relu(self.bn4(self.deconv4(h)))
        h = torch.nn.functional.leaky_relu(self.bn5(self.deconv5(h)))
        h = torch.sigmoid(self.deconv6(h))
        return h

class Large_Discriminator(nn.Module):

    def __init__(self, in_ch):
        super(Large_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2)
        self.fc = nn.Linear(13,1024)
        self.bn4 = nn.BatchNorm2d(3407872)
        if in_ch == 1:
            self.fc4 = nn.Linear(1024, 1)
        else:
            self.fc4 = nn.Linear(3407872, 1)

    def forward(self, x):
        h = torch.nn.functional.leaky_relu(self.bn1(self.conv1(x)))
        h = torch.nn.functional.leaky_relu(self.bn2(self.conv2(h)))
        h = torch.nn.functional.leaky_relu(self.bn3(self.conv3(h)))
        h = torch.nn.functional.leaky_relu(self.bn3(self.conv4(h)))
        h = torch.nn.functional.leaky_relu(self.bn3(self.fc(h)))
        h = torch.sigmoid(self.fc4(h.view(h.size(0), -1)))
        return h