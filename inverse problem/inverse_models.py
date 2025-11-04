import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


################################################################
# Inverse FNO Model - Modified for Material Identification
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class InverseFNO2d(nn.Module):
    """
    Inverse FNO for material identification
    Input: displacement fields + force/BC info (3 channels)
    Output: material mask (segmentation) + optional material properties
    """
    def __init__(self, modes1, modes2, width, num_materials=3, predict_properties=False):
        super(InverseFNO2d, self).__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_materials = num_materials
        self.predict_properties = predict_properties
        self.padding = 9  # pad the domain if input is non-periodic
        
        # Input projection: displacement fields (2) + force/BC (1) + grid coords (2) -> hidden dim
        self.fc0 = nn.Linear(5, self.width)
        
        # Fourier layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        # Skip connections
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        # Batch normalization
        self.bn0 = nn.BatchNorm2d(self.width)
        self.bn1 = nn.BatchNorm2d(self.width)
        self.bn2 = nn.BatchNorm2d(self.width)
        self.bn3 = nn.BatchNorm2d(self.width)
        
        # Output projection for material segmentation
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.num_materials)
        
        # Optional material property prediction heads
        if self.predict_properties:
            self.prop_fc1 = nn.Linear(self.width, 128)
            self.prop_fc2 = nn.Linear(128, 64)
            # Predict 3 properties: Young's modulus, Poisson's ratio, density
            self.prop_fc3 = nn.Linear(64, 3)
            
    def forward(self, x):
        """
        x: [batch, 3, height, width] - (ux, uy, force/BC)
        Returns:
            - material_logits: [batch, num_materials, height, width] for segmentation
            - properties: [batch, 3, height, width] if predict_properties=True
        """
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)  # Add positional encoding
        
        x = x.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [batch, channels, height, width]
        
        # Padding
        x = F.pad(x, [0, self.padding, 0, self.padding])
        
        # Fourier layers with residual connections
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.bn0(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.bn1(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.bn2(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.bn3(x)
        
        # Remove padding
        x = x[..., :-self.padding, :-self.padding]
        
        # Material segmentation head
        x_seg = x.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        x_seg = self.fc1(x_seg)
        x_seg = F.gelu(x_seg)
        material_logits = self.fc2(x_seg)  # [batch, height, width, num_materials]
        material_logits = material_logits.permute(0, 3, 1, 2)  # [batch, num_materials, height, width]
        
        if self.predict_properties:
            # Material property prediction head
            x_prop = x.permute(0, 2, 3, 1)
            x_prop = self.prop_fc1(x_prop)
            x_prop = F.gelu(x_prop)
            x_prop = self.prop_fc2(x_prop)
            x_prop = F.gelu(x_prop)
            properties = self.prop_fc3(x_prop)  # [batch, height, width, 3]
            properties = properties.permute(0, 3, 1, 2)  # [batch, 3, height, width]
            
            return material_logits, properties
        
        return material_logits
    
    def get_grid(self, shape, device):
        """Generate coordinate grid for positional encoding"""
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device).permute(0, 3, 1, 2)


################################################################
# Inverse CNN Model - U-Net style for material identification
################################################################

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class InverseUNet(nn.Module):
    """
    U-Net for inverse material identification
    Input: displacement fields + force/BC (3 channels)
    Output: material segmentation + optional properties
    """
    def __init__(self, n_channels=3, n_classes=3, predict_properties=False, bilinear=False):
        super(InverseUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.predict_properties = predict_properties
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Material segmentation output
        self.outc = OutConv(64, n_classes)
        
        # Material properties output (optional)
        if self.predict_properties:
            self.prop_out = OutConv(64, 3)  # Young's modulus, Poisson's ratio, density

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Material segmentation
        material_logits = self.outc(x)
        
        if self.predict_properties:
            # Material properties
            properties = self.prop_out(x)
            return material_logits, properties
            
        return material_logits
