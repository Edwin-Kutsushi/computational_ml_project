import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        return self.net(x)
    


 ## Baseline CNN Model for MNIST
 # This is a simple CNN architecture that serves as a baseline for our experiments. It consists of two convolutional layers followed by fully connected layers. The model is designed to be straightforward and effective for the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits.

import torch.nn as nn


class CNNModel(nn.Module):

    def __init__(self, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        return self.net(x)    