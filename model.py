import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
import torch.nn.functional as F

# --- Helper Modules ---
class ConvBlock(nn.Module):
    """Standard Convolutional Block: Conv -> BatchNorm -> SiLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))

class DownBlock(nn.Module):
    """Downsampling Block: ConvBlock with stride=2"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=2)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UpBlock(nn.Module):
    """Up-sampling Block with Skip Connection"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After concatenation: (in_channels // 2) + skip_channels
        self.conv1 = ConvBlock(in_channels // 2 + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip_x):
        x = self.up(x) 
        if x.shape[2:] != skip_x.shape[2:]:
            skip_x = F.interpolate(skip_x, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ConvLSTM2d(nn.Module):
    """2D Convolutional LSTM for spatial-temporal features"""
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels,
                            kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x, hidden_state=None):
        b, _, h, w = x.shape
        if hidden_state is None:
            h_state = torch.zeros(b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype)
            c_state = torch.zeros(b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype)
        else:
            h_state, c_state = hidden_state
        
        gates = self.conv(torch.cat([x, h_state], dim=1))
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        c_next = torch.sigmoid(f) * c_state + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        
        return h_next, (h_next, c_next)

# --- Main Components ---
class YOLOFeatureExtractor(nn.Module):
    """Extract multi-scale features from YOLO backbone using hooks"""
    def __init__(self, model_name='yolo11m.pt', freeze=True):
        super().__init__()
        self.model = YOLO(model_name).model
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
    
    def train(self, mode=True):
        self.training = mode
        self.model.eval()
    
    def forward(self, x):
        with torch.no_grad():
            _, features = self.model(x)
        return tuple(features)

    @torch.no_grad()
    def get_feature_channels(self, dummy_input_shape=(1, 3, 640, 640)):
        """Helper to get feature channel counts dynamically."""
        dummy_input = torch.randn(*dummy_input_shape, device=next(self.model.parameters()).device)        
        _, features = self.model(dummy_input)
        return [f.shape[1] for f in features]

class TemporalUNet(nn.Module):
    """U-Net with LSTM bottleneck for temporal modeling"""
    def __init__(self, feature_channels, use_conv_lstm=True):
        super().__init__()
        ch_p3, ch_p4, ch_p5 = feature_channels
        self.use_conv_lstm = use_conv_lstm
        
        # Encoder
        self.enc1, self.down1 = ConvBlock(ch_p3, 128), DownBlock(128, 256)
        self.enc2, self.down2 = ConvBlock(256 + ch_p4, 256), DownBlock(256, 512)
        self.enc3, self.down3 = ConvBlock(512 + ch_p5, 512), DownBlock(512, 1024)
        
        # LSTM Bottleneck
        self.lstm = ConvLSTM2d(1024, 1024) if use_conv_lstm else \
                    nn.LSTM(1024, 1024, num_layers=2, batch_first=True, dropout=0.1)
        self.bottleneck_conv = ConvBlock(1024, 1024)
        
        # Decoder
        self.up1, self.up2, self.up3 = UpBlock(1024, 512, 512), UpBlock(512, 256, 256), UpBlock(256, 128, 128)
        self.out_p5, self.out_p4, self.out_p3 = nn.Conv2d(512, ch_p5, 1), nn.Conv2d(256, ch_p4, 1), nn.Conv2d(128, ch_p3, 1)

    def forward(self, features, hidden_state=None):
        p3, p4, p5 = features
        
        # Encoder with fusion
        x1 = self.enc1(p3)
        x2 = self.enc2(torch.cat([self.down1(x1), p4], dim=1))
        x3 = self.enc3(torch.cat([self.down2(x2), p5], dim=1))
        x = self.down3(x3)
        
        # LSTM Bottleneck
        if self.use_conv_lstm:
            x, new_hidden = self.lstm(x, hidden_state)
        else:
            b, c, h, w = x.shape
            x_flat = x.view(b, c, -1).permute(0, 2, 1)
            x_flat, new_hidden = self.lstm(x_flat, hidden_state)
            x = x_flat.permute(0, 2, 1).view(b, c, h, w)
        
        x = self.bottleneck_conv(x)
        
        # Decoder with outputs
        d1 = self.up1(x, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)
        
        return (self.out_p3(d3), self.out_p4(d2), self.out_p5(d1)), new_hidden


class YOLOTemporalUNet(nn.Module):
    """
    Main model combining YOLO feature extraction with temporal U-Net processing.
    Suitable for video object detection or tracking tasks.
    """
    def __init__(self, num_classes=80, yolo_model_name='yolo11m.pt', 
                 use_conv_lstm=True):
        super(YOLOTemporalUNet, self).__init__()
        
        self.feature_extractor = YOLOFeatureExtractor(model_name=yolo_model_name, freeze=True)
        feature_channels = self.feature_extractor.get_feature_channels()
        self.temporal_unet = TemporalUNet(feature_channels=feature_channels, 
                                         use_conv_lstm=use_conv_lstm)

        self.detection_head = Detect(nc=num_classes, ch=feature_channels)
        self.model = nn.ModuleList([self.detection_head])

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: Input image tensor (B, C, H, W)
            hidden_state: Previous LSTM hidden state for temporal processing
        
        Returns:
            detections: Tuple of detection outputs at different scales
            hidden_state: Updated LSTM hidden state
        """
        yolo_features = self.feature_extractor(x)
        temporal_features, new_hidden = self.temporal_unet(yolo_features, hidden_state)
        detections = self.detection_head(list(temporal_features))

        return detections, new_hidden