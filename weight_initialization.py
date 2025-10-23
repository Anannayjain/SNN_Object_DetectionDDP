import torch
import torch.nn as nn
import math
from model import ConvLSTM2d

def initialize_weights(model):
    """
    Initialize weights for YOLOTemporalUNet model.
    
    Strategy:
    - Conv layers: Kaiming initialization (good for ReLU/SiLU)
    - BatchNorm: weight=1, bias=0
    - LSTM: Xavier uniform for weights, zero for biases
    - Detection head: Special initialization for stability
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Kaiming initialization for convolutional layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.ConvTranspose2d):
            # Kaiming for transposed convolutions (upsampling)
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.BatchNorm2d):
            # Standard BatchNorm initialization
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.LSTM):
            # Xavier initialization for LSTM weights
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Set forget gate bias to 1 for better gradient flow
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)
        
        elif isinstance(m, ConvLSTM2d):
            # Initialize ConvLSTM2d
            nn.init.xavier_uniform_(m.conv.weight)
            if m.conv.bias is not None:
                nn.init.constant_(m.conv.bias, 0)
                # Set forget gate bias to 1
                n = m.conv.bias.size(0)
                m.conv.bias.data[n//4:n//2].fill_(1)

def initialize_detection_head(detection_head, num_classes):
    """
    Special initialization for YOLO detection head.
    Helps with training stability and faster convergence.
    """
    for m in detection_head.modules():
        if isinstance(m, nn.Conv2d):
            # Use normal initialization with smaller std for stability
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # Initialize classification layers with bias trick
    # This helps with class imbalance (more background than objects)
    if hasattr(detection_head, 'cv3'):  # Classification convs
        for conv in detection_head.cv3:
            if hasattr(conv, 'bias') and conv.bias is not None:
                # Prior probability initialization
                # Assumes 1% of anchors have objects initially
                prior_prob = 0.01
                bias_init = -math.log((1 - prior_prob) / prior_prob)
                conv.bias.data.fill_(bias_init)

def initialize_model(model, init_detection_head_special=True):
    """
    Complete initialization for YOLOTemporalUNet model.
    
    Args:
        model: YOLOTemporalUNet instance
        init_detection_head_special: Whether to use special detection head initialization
    
    Returns:
        model: Model with initialized weights
    """
    print("Initializing model weights...")
    
    # Initialize all layers except frozen YOLO backbone
    initialize_weights(model.temporal_unet)
    
    # Special initialization for detection head
    if init_detection_head_special:
        initialize_detection_head(model.detection_head, model.nc)
    else:
        initialize_weights(model.detection_head)
    
    print("Weight initialization complete!")
    return model

# Alternative: Layer-wise learning rate initialization helper
def get_param_groups(model, lr=1e-3, weight_decay=1e-4):
    """
    Create parameter groups with different learning rates.
    Useful for fine-tuning with discriminative learning rates.
    
    Args:
        model: YOLOTemporalUNet instance
        lr: Base learning rate
        weight_decay: Weight decay for regularization
    
    Returns:
        List of parameter groups for optimizer
    """
    # Group 1: Frozen YOLO backbone (no training)
    # Group 2: Temporal U-Net (normal lr)
    # Group 3: Detection head (higher lr for faster convergence)
    
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    
    for k, v in model.named_parameters():
        if not v.requires_grad:
            continue
        
        if 'detection_head' in k:
            if 'bias' in k:
                pg2.append(v)  # biases (no decay)
            else:
                pg1.append(v)  # detection head weights
        elif 'temporal_unet' in k:
            if 'bias' in k or 'bn' in k:
                pg2.append(v)  # biases and BN (no decay)
            else:
                pg0.append(v)  # temporal unet weights
    
    return [
        {'params': pg0, 'lr': lr, 'weight_decay': weight_decay},
        {'params': pg1, 'lr': lr * 2, 'weight_decay': weight_decay},  # detection head gets higher lr
        {'params': pg2, 'lr': lr, 'weight_decay': 0.0}  # no decay for biases/BN
    ]