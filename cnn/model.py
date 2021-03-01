import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from conv_net import ConvNet

class Model():

    def __init__(self, num_epochs=5, num_classes=10, batch_size=100, learning_rate=0.001):

        self.model = ConvNet(num_classes)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, train_loader):

        total_step = len(train_loader)

        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, self.num_epochs, i+1, total_step, loss.item()))

    def eval(self, test_loader):

        self.model.eval()

        with torch.no_grad():

            correct = 0
            total = 0

            for images, labels in test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    def save(self):
        # Save the model checkpoint
        torch.save(self.model.state_dict(), 'model.ckpt')