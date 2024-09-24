import torch
import torch.nn    as nn
import torchvision as tv
import torch.nn.functional as F

class BasicConvolution(nn.Module):
    def __init__(self, input_ch : int, output_ch : int) -> None:
        super().__init__()

        self.convolution   = nn.Conv2d(input_ch, output_ch, kernel_size = 3, padding = 'same')
        self.normalization = nn.BatchNorm2d(output_ch)
        self.activation    = nn.LeakyReLU()
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x

class BasicConvBlock(nn.Module):
    def __init__(self, input_ch : int, output_ch : int) -> None:
        super().__init__()
        self.conv = BasicConvolution(input_ch, output_ch)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, output_class : int) -> None:
        super().__init__()

        # 3, 128, 128
        self.conv_1  = nn.Conv2d(3, 16, kernel_size = 7, padding = "same")
        self.block_1 = BasicConvBlock(16, 32)
        
        # 32, 64, 64
        self.block_2 = BasicConvBlock(32, 64)
        
        # 64, 32, 32
        self.block_3 = BasicConvBlock(64, 128)

        # 128,16, 16
        self.block_4 = BasicConvBlock(128, 256)

        # 256, 8,  8
        self.pool    = nn.AdaptiveMaxPool2d(1)

        # 256, 1, 1
        # RESHAPE 
        # 256

        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_class)
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)  # (N,   3, 128, 128) => (N,  16, 128, 128)
        
        x = self.block_1(x) # (N,  16, 128, 128) => (N,  32,  64,  64)
        x = self.block_2(x) # (N,  32,  64,  64) => (N,  64,  32,  32)
        x = self.block_3(x) # (N,  64,  32,  32) => (N,  64,  16,  16)
        x = self.block_4(x) # (N,  64,  16,  16) => (N,  64,   8,   8)
        
        x = self.pool(x)    # (N, 128,   8,   8) => (N, 128,   1,   1)
        
        # RESHAPE OPERATION
        bz = x.size(0)      # batch   size
        cz = x.size(1)      # channel size 
        x  = x.view(bz, cz) # (N, 128,   1,   1) => (N, 128)  ## 4D Tensor => 2D Tensor
        
        x = self.head(x)    # (N, 128)           => (N, output_class)
        return x

class BasicMobileNet(nn.Module):
    def __init__(self, output_classes : int) -> None:
        super().__init__()

        self.base = tv.models.mobilenet_v3_small(weights = tv.models.MobileNet_V3_Small_Weights.DEFAULT)
        self.base.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, output_classes)
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        return x

if __name__ == "__main__":
    print("Model Base Run")

    t     = torch.rand(1, 3, 64, 64)
    model = BasicMobileNet(7)
    y     = model(t)
    print(y)
