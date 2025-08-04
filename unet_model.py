import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        
        # if bilinear, use the normal convolutions to reduce the number of channels
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

class ColorEmbedding(nn.Module):
    """Embedding for color names"""
    def __init__(self, num_colors, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_colors, embed_dim)
        
    def forward(self, color_ids):
        return self.embedding(color_ids)

class ConditionalUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, num_colors=10, color_embed_dim=64, bilinear=False):
        super(ConditionalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.color_embed_dim = color_embed_dim
        
        # Color embedding
        self.color_embedding = ColorEmbedding(num_colors, color_embed_dim)
        
        # UNet encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Color conditioning layers - inject color info at bottleneck
        self.color_proj = nn.Sequential(
            nn.Linear(color_embed_dim, 1024 // factor),
            nn.ReLU(),
            nn.Linear(1024 // factor, 1024 // factor)
        )
        
        # UNet decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
    def forward(self, x, color_ids):
        # Encode input image
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Get color embedding and condition the bottleneck
        color_emb = self.color_embedding(color_ids)  # [batch_size, color_embed_dim]
        color_features = self.color_proj(color_emb)  # [batch_size, 1024//factor]
        
        # Reshape color features to match spatial dimensions and add to bottleneck
        batch_size, channels = color_features.shape
        spatial_size = x5.shape[2]  # Assuming square feature maps
        color_features = color_features.view(batch_size, channels, 1, 1)
        color_features = color_features.expand(-1, -1, spatial_size, spatial_size)
        
        # Add color conditioning to bottleneck features
        x5 = x5 + color_features
        
        # Decode
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        # Apply sigmoid to get output in [0, 1] range
        return torch.sigmoid(logits)

# Color mapping utility
class ColorMapper:
    def __init__(self):
        self.color_to_id = {
            'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'purple': 4,
            'orange': 5, 'pink': 6, 'brown': 7, 'black': 8, 'white': 9
        }
        self.id_to_color = {v: k for k, v in self.color_to_id.items()}
    
    def color_name_to_id(self, color_name):
        return self.color_to_id.get(color_name.lower(), 0)  # Default to red if unknown
    
    def id_to_color_name(self, color_id):
        return self.id_to_color.get(color_id, 'red')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConditionalUNet(n_channels=3, n_classes=3, num_colors=10).to(device)
    
    print(f"Model has {count_parameters(model):,} trainable parameters")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256).to(device)
    color_ids = torch.randint(0, 10, (batch_size,)).to(device)
    
    with torch.no_grad():
        output = model(x, color_ids)
        print(f"Input shape: {x.shape}")
        print(f"Color IDs: {color_ids}")
        print(f"Output shape: {output.shape}")
        print("Model test passed!")
