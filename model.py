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
        
        # Handle spatial dimension mismatch
        if x.shape[2:] != skip_x.shape[2:]:
            x = nn.functional.interpolate(x, size=skip_x.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvLSTM2d(nn.Module):
    """2D Convolutional LSTM for spatial-temporal features"""
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super(ConvLSTM2d, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # Gates: input, forget, cell, output
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, hidden_state=None):
        batch_size, _, height, width = x.shape
        
        if hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_channels, height, width, 
                          device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_channels, height, width,
                          device=x.device, dtype=x.dtype)
        else:
            h, c = hidden_state
        
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, (h_next, c_next)


# --- Main Components ---

class YOLOFeatureExtractor(nn.Module):
    """
    Extract multi-scale features from a YOLO backbone.
    This is a more robust implementation that does not rely on hardcoded layer indices.
    """
    def __init__(self, model_name='yolov8n.pt', freeze=True):
        super(YOLOFeatureExtractor, self).__init__()
        
        # Load a pretrained YOLO model
        yolo = YOLO(model_name)
        
        # We use the 'model.model' which is the actual neural network part
        self.model = yolo.model
        
        # Define which layers to save for feature extraction.
        # These correspond to the end of C2, C3, C4, and C5 stages in YOLOv8.
        # Adjust these indices based on the specific model's architecture if needed,
        # but these are generally stable for YOLOv8.
        self.model.model[4].save = True  # C2, stride 8
        self.model.model[6].save = True  # C3, stride 16
        self.model.model[9].save = True  # C4, C5, stride 32 (Detect() head)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(self, x):
        """
        Performs a forward pass and returns the extracted features.
        """
        # The ultralytics model forward pass can directly return features
        # from the layers we marked with `save=True`.
        # The result is a list of feature maps.
        features = self.model(x)
        
        # For the layers we marked, the features will be:
        # features[0] -> from layer 4 (P3/C3 in some contexts)
        # features[1] -> from layer 6 (P4/C4 in some contexts)
        # features[2] -> from layer 9 (P5/C5 in some contexts, part of the detection head)
        
        # Ensure we have enough features before trying to access them
        if len(features) < 3:
            raise ValueError(f"Expected at least 3 feature maps, but got {len(features)}. Check the 'save' attributes.")

        p3, p4, p5 = features[0], features[1], features[2]
        
        return p3, p4, p5


class TemporalUNet(nn.Module):
    """
    U-Net architecture with temporal modeling via LSTM at bottleneck.
    Takes multi-scale YOLO features and processes them through encoder-decoder.
    """
    def __init__(self, feature_channels, use_conv_lstm=True):
        super(TemporalUNet, self).__init__()
        ch_p3, ch_p4, ch_p5 = feature_channels 
        self.use_conv_lstm = use_conv_lstm
        
        # --- Encoder Path (Downsampling) ---
        # Start from P3 and downsample
        self.enc1 = ConvBlock(ch_p3, 128)
        self.down1 = DownBlock(128, 256)  # Downsample to match P4 size
        
        # Fusion with P4
        self.enc2 = ConvBlock(256 + ch_p4, 256)
        self.down2 = DownBlock(256, 512)  # Downsample to match P5 size
        
        # Fusion with P5
        self.enc3 = ConvBlock(512 + ch_p5, 512)
        self.down3 = DownBlock(512, 1024)  # Bottleneck
        
        # --- Bottleneck with LSTM ---
        if use_conv_lstm:
            # ConvLSTM2d preserves spatial structure
            self.lstm = ConvLSTM2d(in_channels=1024, hidden_channels=1024, kernel_size=3)
        else:
            # Standard LSTM flattens spatial dimensions
            self.lstm = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, 
                              batch_first=True, dropout=0.1)
        
        self.bottleneck_conv = ConvBlock(1024, 1024)
        
        # --- Decoder Path (Upsampling) with Skip Connections ---
        self.up1 = UpBlock(in_channels=1024, skip_channels=512, out_channels=512)
        self.up2 = UpBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.up3 = UpBlock(in_channels=256, skip_channels=128, out_channels=128)
        
        # Output feature projections matching YOLO feature sizes
        self.out_p5 = nn.Conv2d(512, ch_p5, kernel_size=1)
        self.out_p4 = nn.Conv2d(256, ch_p4, kernel_size=1)
        self.out_p3 = nn.Conv2d(128, ch_p3, kernel_size=1)

    def forward(self, features, hidden_state=None):
        p3, p4, p5 = features  # Multi-scale YOLO features
        
        # --- Encoder Path with Feature Fusion ---
        # Process P3
        x1 = self.enc1(p3)
        x = self.down1(x1)
        
        # Fuse with P4
        x = torch.cat([x, p4], dim=1)
        x2 = self.enc2(x)
        x = self.down2(x2)
        
        # Fuse with P5
        x = torch.cat([x, p5], dim=1)
        x3 = self.enc3(x)
        x = self.down3(x3)
        
        # --- Bottleneck with Temporal Modeling ---
        if self.use_conv_lstm:
            x, new_hidden = self.lstm(x, hidden_state)
        else:
            b, c, h, w = x.shape
            # Flatten spatial dimensions for standard LSTM
            x_flat = x.view(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)
            x_flat, new_hidden = self.lstm(x_flat, hidden_state)
            x = x_flat.permute(0, 2, 1).view(b, c, h, w)
        
        x = self.bottleneck_conv(x)
        
        # --- Decoder Path with Skip Connections ---
        d1 = self.up1(x, x3)      # Skip from enc3
        out_p5 = self.out_p5(d1)  # Output feature at P5 scale
        
        d2 = self.up2(d1, x2)     # Skip from enc2
        out_p4 = self.out_p4(d2)  # Output feature at P4 scale
        
        d3 = self.up3(d2, x1)     # Skip from enc1
        out_p3 = self.out_p3(d3)  # Output feature at P3 scale
        
        return (out_p3, out_p4, out_p5), new_hidden


class YOLOTemporalUNet(nn.Module):
    """
    Main model combining YOLO feature extraction with temporal U-Net processing.
    Suitable for video object detection or tracking tasks.
    """
    def __init__(self, num_classes=80, yolo_model_name='yolo11n.pt', 
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
        yolo_model_name='yolo11n.pt',
        feature_channels=(256, 192, 384),  # Adjusted feature channels based on extracted features
        use_conv_lstm=True  # Set to False to use standard LSTM
    ).to(device)
    
    dummy_input = torch.randn(2, 3, 640, 640).to(device)
    
    outputs1, hidden1 = model(dummy_input, hidden_state=None)

    # print("\nPerforming test forward pass...")
    # try:
    #     with torch.no_grad():
    #         # First frame (no hidden state)
    #         outputs1, hidden1 = model(dummy_input, hidden_state=None)
            
    #         # Second frame (with hidden state for temporal continuity)
    #         outputs2, hidden2 = model(dummy_input, hidden_state=hidden1)
        
    #     print(f"\nInput shape: {dummy_input.shape}")
    #     print(f"Output P3 shape: {outputs1[0].shape}")
    #     print(f"Output P4 shape: {outputs1[1].shape}")
    #     print(f"Output P5 shape: {outputs1[2].shape}")
    #     print("\nModel test successful!")
        
    #     # Count parameters
    #     total_params = sum(p.numel() for p in model.parameters())
    #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     print(f"\nTotal parameters: {total_params:,}")
    #     print(f"Trainable parameters: {trainable_params:,}")
        
    # except Exception as e:
    #     print(f"\nError during forward pass: {e}")
    #     print("\nTrying to extract YOLO features to determine correct feature channels...")
        
    #     try:
    #         with torch.no_grad():
    #             p3, p4, p5 = model.feature_extractor(dummy_input)
    #             print(f"\nDetected YOLO feature channels:")
    #             print(f"P3: {p3.shape}")
    #             print(f"P4: {p4.shape}")
    #             print(f"P5: {p5.shape}")
    #             print(f"\nPlease reinitialize the model with:")
    #             print(f"feature_channels=({p3.shape[1]}, {p4.shape[1]}, {p5.shape[1]})")
    #     except Exception as e2:
    #         print(f"Error extracting features: {e2}")
    #         print("\nPlease check your YOLO model version and adjust feature extraction indices.")