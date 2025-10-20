import torch
import torch.nn as nn
from ultralytics import YOLO

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
    def __init__(self, model_name='yolo11m.pt', freeze=True, feature_layers=[15, 18, 21]):
        super().__init__()
        self.model = YOLO(model_name).model
        self.features = {}
        self.feature_layers = feature_layers
        self.hooks = [self.model.model[i].register_forward_hook(self._make_hook(i)) 
                     for i in feature_layers]
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
    
    def _make_hook(self, layer_id):
        return lambda m, i, o: self.features.update({layer_id: o})
    
    def forward(self, x):
        self.features.clear()
        _ = self.model(x)
        return tuple(self.features[i] for i in self.feature_layers)
    
    def __del__(self):
        for hook in self.hooks:
            hook.remove()

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
                 feature_channels=None, use_conv_lstm=True):
        super(YOLOTemporalUNet, self).__init__()
        
        self.feature_extractor = YOLOFeatureExtractor(model_name=yolo_model_name, freeze=True)
        self.temporal_unet = TemporalUNet(feature_channels=feature_channels, 
                                         use_conv_lstm=use_conv_lstm)
        
        # Detection heads for each scale
        self.head_p3 = nn.Conv2d(feature_channels[0], num_classes + 5, kernel_size=1)
        self.head_p4 = nn.Conv2d(feature_channels[1], num_classes + 5, kernel_size=1)
        self.head_p5 = nn.Conv2d(feature_channels[2], num_classes + 5, kernel_size=1)

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
        
        out_p3, out_p4, out_p5 = temporal_features
        det_p3 = self.head_p3(out_p3)
        det_p4 = self.head_p4(out_p4)
        det_p5 = self.head_p5(out_p5)
        
        return (det_p3, det_p4, det_p5), new_hidden

# --- Example Usage ---
if __name__ == '__main__':
    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("WARNING: Running on CPU. This will be very slow.")
    
    # Initialize model
    model = YOLOTemporalUNet(
        num_classes=80,
        yolo_model_name='yolo11m.pt',
        feature_channels=(1024, 768, 1024),  # Adjusted feature channels based on extracted features
        use_conv_lstm=True  # Set to False to use standard LSTM
    ).to(device)
    
    dummy_input = torch.randn(2, 3, 640, 640).to(device)
    
    # outputs1, hidden1 = model(dummy_input, hidden_state=None)

    print("\nPerforming test forward pass...")
    try:
        with torch.no_grad():
            # First frame (no hidden state)
            outputs1, hidden1 = model(dummy_input, hidden_state=None)
            
            # Second frame (with hidden state for temporal continuity)
            outputs2, hidden2 = model(dummy_input, hidden_state=hidden1)
        
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Output P3 shape: {outputs1[0].shape}")
        print(f"Output P4 shape: {outputs1[1].shape}")
        print(f"Output P5 shape: {outputs1[2].shape}")
        print("\nModel test successful!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"\nError during forward pass: {e}")
        print("\nTrying to extract YOLO features to determine correct feature channels...")
        
        try:
            with torch.no_grad():
                p3, p4, p5 = model.feature_extractor(dummy_input)
                print(f"\nDetected YOLO feature channels:")
                print(f"P3: {p3.shape}")
                print(f"P4: {p4.shape}")
                print(f"P5: {p5.shape}")
                print(f"\nPlease reinitialize the model with:")
                print(f"feature_channels=({p3.shape[1]}, {p4.shape[1]}, {p5.shape[1]})")
        except Exception as e2:
            print(f"Error extracting features: {e2}")
            print("\nPlease check your YOLO model version and adjust feature extraction indices.")