import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import os
from pathlib import Path
import random
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from collections import defaultdict

# --- Import your custom modules ---
from model import YOLOTemporalUNet
from dataset import DSECDataset

from ultralytics.utils.loss import v8DetectionLoss

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, sequence_length):
    """
    Performs one full epoch of training.
    """
    model.train()
    total_loss = 0.0
    
    # Using tqdm for a progress bar
    pbar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (image_tensor, labels_tensor) in enumerate(pbar):
        # image_tensor shape: (B, S, C, H, W)
        # labels_tensor shape: (B, N, 5) where N is number of objects
        
        image_tensor = image_tensor.to(device)
        labels_tensor = labels_tensor.to(device)
        
        # Reset gradients for each new batch
        optimizer.zero_grad()
        
        # Initialize hidden state for the LSTM at the start of each sequence
        hidden_state = None
        
        # Loop through the time sequence
        for t in range(sequence_length):
            frame = image_tensor[:, t, :, :, :]
            preds, hidden_state = model(frame, hidden_state)

        # The loss is calculated only on the predictions from the final frame
        loss, _ = loss_fn(preds, labels_tensor)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar description
        pbar.set_postfix(loss=f'{loss.item():.4f}')
        
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, loss_fn, device, sequence_length):
    """
    Performs one full epoch of validation.
    """
    model.eval()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for batch_idx, (image_tensor, labels_tensor) in enumerate(pbar):
            image_tensor = image_tensor.to(device)
            labels_tensor = labels_tensor.to(device)
            
            hidden_state = None
            
            for t in range(sequence_length):
                frame = image_tensor[:, t, :, :, :]
                preds, hidden_state = model(frame, hidden_state)
                
            loss, _ = loss_fn(preds, labels_tensor)
            total_loss += loss.item()
            
            pbar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / len(dataloader)


if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    """
    Main function to run the training and validation process.
    """
    set_seed(config['training']['seed'])
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directory for saving runs
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Data Loading ---
    print("Loading datasets...")
    full_train_dataset = DSECDataset(config, mode="train")

    # Group indices by sequence
    seq_groups = defaultdict(list)
    for idx, (img_dir, _, _) in enumerate(full_train_dataset.samples):
        seq_groups[str(img_dir)].append(idx)

    # Split and build indices efficiently
    train_seqs, val_seqs = train_test_split(list(seq_groups), test_size=0.2, random_state=42)
    train_seqs, val_seqs = set(train_seqs), set(val_seqs)  # O(1) lookup

    train_indices, val_indices = [], []
    for seq, indices in seq_groups.items():
        (train_indices if seq in train_seqs else val_indices).extend(indices)

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # --- Model Initialization ---
    print("Initializing model...")
    model_config = config['model']
    model = YOLOTemporalUNet(
        num_classes=model_config['num_classes'],
        yolo_model_name=model_config['yolo_model_name'],
        feature_channels=tuple(model_config['feature_channels']),
        use_conv_lstm=model_config['use_conv_lstm']
    ).to(device)
    
    # --- Loss Function, Optimizer ---
    # The v8DetectionLoss class from ultralytics handles the complex
    # box, class, and objectness losses for you.
    loss_fn = v8DetectionLoss(model)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # --- Training Loop ---
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        
        train_loss = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            loss_fn, 
            device, 
            config['dataset']['train']['seq_len']
        )
        print(f"Average Training Loss: {train_loss:.4f}")
        
        val_loss = validate_one_epoch(
            model, 
            val_loader, 
            loss_fn, 
            device, 
            config['dataset']['val']['seq_len']
        )
        print(f"Average Validation Loss: {val_loss:.4f}")
        
        # --- Save Checkpoints ---
        # Save the latest model
        latest_checkpoint_path = save_dir / "latest.pt"
        torch.save(model.state_dict(), latest_checkpoint_path)
        print(f"Saved latest model checkpoint to {latest_checkpoint_path}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = save_dir / "best.pt"
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"New best model saved to {best_checkpoint_path} with validation loss: {best_val_loss:.4f}")

    print("\nTraining finished!")

