import torch
import torch.nn as nn
import math
from model import ConvLSTM2d
# --- ADDED THIS IMPORT ---
from ultralytics.nn.modules.head import Detect 

def initialize_weights(m):
    """
    Recursive weight initialization function to be applied to modules.
    
    Strategy:
    - Conv layers: Kaiming initialization (good for ReLU/SiLU)
    - BatchNorm: weight=1, bias=0
    - LSTM: Xavier uniform for weights, zero for biases (with forget gate trick)
    """
    # --- MODIFIED: This function is now designed to be used with .apply() ---
    
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

# --- DELETED ---
# The entire `initialize_detection_head` function has been removed.
# It was conflicting with the Detect module's internal, correct initialization.

def initialize_model(model):
    """
    Complete initialization for YOLOTemporalUNet model.
    
    Args:
        model: YOLOTemporalUNet instance
    
    Returns:
        model: Model with initialized weights
    """
    # --- MODIFIED: This function is now much simpler ---
    print("Initializing model weights...")
    
    # Initialize all layers except frozen YOLO backbone and Detect head
    print("Applying custom initialization to model.temporal_unet...")
    model.temporal_unet.apply(initialize_weights)
    
    print("Skipping model.feature_extractor (frozen).")
    print("Skipping model.detection_head (it self-initializes correctly).")
    
    print("Weight initialization complete!")
    return model

# --- KEPT AS-IS: This is a useful helper for your optimizer ---
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