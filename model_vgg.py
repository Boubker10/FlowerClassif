import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConvolution(nn.Module):
    def __init__(self, input_ch: int, output_ch: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(input_ch, output_ch, kernel_size=3, padding='same')
        self.normalization = nn.BatchNorm2d(output_ch)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x

class BasicConvBlock(nn.Module):
    def __init__(self, input_ch: int, output_ch: int) -> None:
        super().__init__()
        self.conv1 = BasicConvolution(input_ch, output_ch)
        self.conv2 = BasicConvolution(output_ch, output_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x

class VGGCustom(nn.Module):
    def __init__(self, output_classes: int) -> None:
        super().__init__()

        # VGG blocks (conv + pool)
        self.block1 = BasicConvBlock(3, 64)
        self.block2 = BasicConvBlock(64, 128)
        self.block3 = BasicConvBlock(128, 256)
        self.block4 = BasicConvBlock(256, 512)

        # Adaptive pooling to ensure fixed output size
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, output_classes)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional layers with pooling
        x = self.block1(x)  # (N, 3, 224, 224) -> (N, 64, 112, 112)
        x = self.block2(x)  # (N, 64, 112, 112) -> (N, 128, 56, 56)
        x = self.block3(x)  # (N, 128, 56, 56) -> (N, 256, 28, 28)
        x = self.block4(x)  # (N, 256, 28, 28) -> (N, 512, 14, 14)
        x = self.pool(x)    # Adaptive pooling -> (N, 512, 7, 7)

        # Flattening
        x = x.view(x.size(0), -1)  # Flatten (N, 512*7*7)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))    # (N, 512*7*7) -> (N, 4096)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))    # (N, 4096) -> (N, 4096)
        x = self.dropout2(x)
        x = self.fc3(x)            # (N, 4096) -> (N, output_classes)

        return x

if __name__ == "__main__":
    # Test the model
    model = VGGCustom(output_classes=2)  # For male/female classification
    x = torch.rand(1, 3, 224, 224)  # Example input
    y = model(x)
    print(y)
