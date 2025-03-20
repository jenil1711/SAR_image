import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.attention(x)

class SARColorizer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = self._make_layer(1, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)
        
        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            AttentionBlock(1024),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.dec4 = self._make_layer(1024, 256)
        self.dec3 = self._make_layer(512, 128)
        self.dec2 = self._make_layer(256, 64)
        self.dec1 = self._make_layer(128, 64)
        
        # Output
        self.final = nn.Conv2d(64, 3, 1)
        
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            AttentionBlock(out_channels)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        
        # Bridge
        b = self.bridge(nn.MaxPool2d(2)(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([nn.Upsample(scale_factor=2)(b), e4], dim=1))
        d3 = self.dec3(torch.cat([nn.Upsample(scale_factor=2)(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([nn.Upsample(scale_factor=2)(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([nn.Upsample(scale_factor=2)(d2), e1], dim=1))
        
        return torch.tanh(self.final(d1))