import torch 
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, num_classes = 10):
        super(CNN, self).__init__()
        # Add sequential layers: 2 Conv and 1 FC
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)
        return output