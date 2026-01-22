
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.PReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.PReLU()
        )

    def forward(self, x):
        return self.net(x)

class UNet1D(nn.Module):
    """Prosty U-Net 1D używany jako model poprawy jakości mowy."""

    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.down1 = nn.Conv1d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1)

        self.enc2 = ConvBlock(base_channels*2, base_channels*2)
        self.down2 = nn.Conv1d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels*4, base_channels*4)

        # Decoder
        self.up1  = nn.ConvTranspose1d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_channels*4, base_channels*2)

        self.up2  = nn.ConvTranspose1d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_channels*2, base_channels)

        self.out  = nn.Conv1d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        b = self.bottleneck(d2)

        u1 = self.up1(b)
        u1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(u1)

        u2 = self.up2(d1)
        u2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(u2)

        out = self.out(d2)
        return out
