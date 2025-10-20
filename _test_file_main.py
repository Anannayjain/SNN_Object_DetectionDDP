import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import yaml
from tqdm import tqdm
import os
from pathlib import Path
import random
import numpy as np
import logging
from datetime import datetime
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from collections import defaultdict

# --- Import your custom modules ---
from model import YOLOTemporalUNet
from dataset import DSECDataset

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.utils import ops

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

def setup_logging(save_dir):
    """Setup logging configuration."""
    log_file = save_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class YOLOLossWrapper(nn.Module):
    """
    Wrapper for v8DetectionLoss to work with custom models.
    Creates a mock model object with required attributes.
    """
    def __init__(self, model, num_classes=2):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        
        # Create a namespace object to hold YOLO args
        class Args:
            def __init__(self):
                self.box = 7.5
                self.cls = 0.5
                self.dfl = 1.5
                self.pose = 12.0
                self.kobj = 2.0
                self.nbs = 64
        
        # Add required attributes for v8DetectionLoss
        self.args = Args()
        self.nc = num_classes
        self.reg_max = 16
        self.stride = torch.tensor([8, 16, 32])  # P3, P4, P5 strides
        self.device = next(model.parameters()).device
        
        # Initialize the actual loss
        self.loss_fn = v8DetectionLoss(self)
        
    def forward(self, preds, batch):
        """
        Args:
            preds: Tuple of (det_p3, det_p4, det_p5)
            batch: Dictionary with 'cls', 'bboxes', 'batch_idx'
        """
        return self.loss_fn(preds, batch)

def format_batch_for_loss(labels_list, device):
    """
    Format labels for Ultralytics loss function.
    
    Args:
        labels_list: List of label tensors, each of shape (N, 5) [class_id, x, y, w, h]
        device: torch device
        
    Returns:
        Dictionary with 'cls', 'bboxes', 'batch_idx' formatted for loss function
    """
    all_cls = []
    all_bboxes = []
    all_batch_idx = []
    
    for batch_idx, labels in enumerate(labels_list):
        if labels.shape[0] == 0:  # No objects
            continue
        
        num_objects = labels.shape[0]
        
        # Extract class IDs and bounding boxes
        cls = labels[:, 0:1]  # (N, 1)
        bboxes = labels[:, 1:5]  # (N, 4) [x, y, w, h]
        
        # Create batch indices
        batch_indices = torch.full((num_objects, 1), batch_idx, 
                                   dtype=torch.float32, device=device)
        
        all_cls.append(cls)
        all_bboxes.append(bboxes)
        all_batch_idx.append(batch_indices)
    
    # Concatenate all
    if len(all_cls) == 0:
        # No labels in entire batch
        return {
            'cls': torch.zeros((0, 1), device=device),
            'bboxes': torch.zeros((0, 4), device=device),
            'batch_idx': torch.zeros((0, 1), device=device)
        }
    
    return {
        'cls': torch.cat(all_cls, dim=0),
        'bboxes': torch.cat(all_bboxes, dim=0),
        'batch_idx': torch.cat(all_batch_idx, dim=0)
    }

def collate_fn(batch):
    """Custom collate function to handle variable number of boxes."""
    images = []
    labels = []
    
    for img, label in batch:
        images.append(img)
        labels.append(label)
    
    images = torch.stack(images, dim=0)
    return images, labels

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, sequence_length, 
                   scaler, epoch, logger, max_grad_norm=10.0):
    """
    Performs one full epoch of training with mixed precision.
    """
    model.train()
    total_loss = 0.0
    loss_items_sum = torch.zeros(3, device=device)  # [box, cls, dfl]
    num_batches = 0
    
    # Using tqdm for a progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch_idx, (image_tensor, labels_list) in enumerate(pbar):
        # image_tensor shape: (B, S, C, H, W)
        # labels_list: list of tensors, each of shape (N, 5)
        
        B, S, C, H, W = image_tensor.shape
        image_tensor = image_tensor.to(device)
        
        # Move labels to device
        labels_list = [label.to(device) for label in labels_list]
        
        # Reset gradients for each new batch
        optimizer.zero_grad()
        
        # Initialize hidden state for the LSTM at the start of each sequence
        hidden_state = None
        
        # Loop through the time sequence
        with autocast():
            for t in range(sequence_length):
                frame = image_tensor[:, t, :, :, :]
                preds, hidden_state = model(frame, hidden_state)
                
                # Detach hidden state to prevent backprop through entire sequence
                if hidden_state is not None:
                    if isinstance(hidden_state, tuple):
                        hidden_state = tuple(h.detach() for h in hidden_state)
                    else:
                        hidden_state = hidden_state.detach()
            
            # Format labels for Ultralytics loss
            batch_dict = format_batch_for_loss(labels_list, device)
            
            # Calculate loss on final frame predictions
            loss, loss_items = loss_fn(preds, batch_dict)
        
        # Check for NaN loss
        if not torch.isfinite(loss):
            logger.warning(f"Loss is {loss.item()}, skipping batch {batch_idx}")
            continue
        
        # Backpropagation with mixed precision
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Accumulate loss items
        loss_items_sum += loss_items.detach()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'box': f'{loss_items[0].item():.4f}',
            'cls': f'{loss_items[1].item():.4f}',
            'dfl': f'{loss_items[2].item():.4f}'
        })
    
    if num_batches == 0:
        logger.warning("No valid batches in training!")
        return 0.0, torch.zeros(3)
    
    avg_loss = total_loss / num_batches
    avg_loss_items = loss_items_sum / num_batches
    
    logger.info(f"Train - Loss: {avg_loss:.4f} | "
                f"Box: {avg_loss_items[0]:.4f} | "
                f"Cls: {avg_loss_items[1]:.4f} | "
                f"DFL: {avg_loss_items[2]:.4f}")
    
    return avg_loss, avg_loss_items

def validate_one_epoch(model, dataloader, loss_fn, device, sequence_length, epoch, logger):
    """
    Performs one full epoch of validation.
    """
    model.eval()
    total_loss = 0.0
    loss_items_sum = torch.zeros(3, device=device)
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
    
    with torch.no_grad():
        for batch_idx, (image_tensor, labels_list) in enumerate(pbar):
            B, S, C, H, W = image_tensor.shape
            image_tensor = image_tensor.to(device)
            labels_list = [label.to(device) for label in labels_list]
            
            hidden_state = None
            
            with autocast():
                for t in range(sequence_length):
                    frame = image_tensor[:, t, :, :, :]
                    preds, hidden_state = model(frame, hidden_state)
                    
                    if hidden_state is not None:
                        if isinstance(hidden_state, tuple):
                            hidden_state = tuple(h.detach() for h in hidden_state)
                        else:
                            hidden_state = hidden_state.detach()
                
                # Format labels
                batch_dict = format_batch_for_loss(labels_list, device)
                
                loss, loss_items = loss_fn(preds, batch_dict)
            
            if not torch.isfinite(loss):
                logger.warning(f"Validation loss is {loss.item()}, skipping batch {batch_idx}")
                continue
            
            total_loss += loss.item()
            num_batches += 1
            loss_items_sum += loss_items.detach()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'box': f'{loss_items[0].item():.4f}',
                'cls': f'{loss_items[1].item():.4f}',
                'dfl': f'{loss_items[2].item():.4f}'
            })
    
    if num_batches == 0:
        logger.warning("No valid batches in validation!")
        return float('inf'), torch.zeros(3)
    
    avg_loss = total_loss / num_batches
    avg_loss_items = loss_items_sum / num_batches
    
    logger.info(f"Val   - Loss: {avg_loss:.4f} | "
                f"Box: {avg_loss_items[0]:.4f} | "
                f"Cls: {avg_loss_items[1]:.4f} | "
                f"DFL: {avg_loss_items[2]:.4f}")
    
    return avg_loss, avg_loss_items

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, 
                   save_path, config, logger):
    """Save model checkpoint with all training state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict(),
        'val_loss': val_loss,
        'config': config,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'user': 'Resnick28'
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device, logger):
    """Load checkpoint and restore training state."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['val_loss']
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    logger.info(f"Checkpoint saved by: {checkpoint.get('user', 'Unknown')}")
    logger.info(f"Checkpoint timestamp: {checkpoint.get('timestamp', 'Unknown')}")
    
    return start_epoch, best_val_loss

def main(resume_from=None):
    """
    Main function to run the training and validation process.
    """
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # --- Setup ---
    set_seed(config['training']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create directory for saving runs
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(save_dir)
    logger.info("="*80)
    logger.info(f"Training started by: Resnick28")
    logger.info(f"Training start time: 2025-10-20 05:06:44 UTC")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    logger.info("="*80)
    
    # --- Data Loading ---
    logger.info("Loading datasets...")
    full_train_dataset = DSECDataset(config, mode="train")

    # Group indices by sequence (assuming samples from same sequence are contiguous)
    sequence_groups = {}
    for idx in range(len(full_train_dataset)):
        image_dir, _, _ = full_train_dataset.samples[idx]
        seq_name = str(image_dir)
        if seq_name not in sequence_groups:
            sequence_groups[seq_name] = []
        sequence_groups[seq_name].append(idx)

    # Split sequences (not individual samples)
    sequence_names = list(sequence_groups.keys())
    train_seqs, val_seqs = train_test_split(
        sequence_names,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # Collect all indices from train and val sequences
    train_indices = []
    for seq in train_seqs:
        train_indices.extend(sequence_groups[seq])

    val_indices = []
    for seq in val_seqs:
        val_indices.extend(sequence_groups[seq])

    # Create subset datasets
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if config['training']['num_workers'] > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if config['training']['num_workers'] > 0 else False
    )
    
    # --- Model Initialization ---
    logger.info("Initializing model...")
    model_config = config['model']
    model = YOLOTemporalUNet(
        num_classes=model_config['num_classes'],
        yolo_model_name=model_config['yolo_model_name'],
        feature_channels=tuple(model_config['feature_channels']),
        use_conv_lstm=model_config['use_conv_lstm']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # --- Loss Function with Wrapper ---
    logger.info("Initializing loss function...")
    loss_fn = YOLOLossWrapper(model, num_classes=model_config['num_classes'])
    
    # --- Optimizer, Scheduler ---
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['learning_rate'] * 0.01
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # --- Load checkpoint if resuming ---
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from:
        start_epoch, best_val_loss = load_checkpoint(
            resume_from, model, optimizer, scheduler, scaler, device, logger
        )
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # --- Training Loop ---
    logger.info("Starting training loop...")
    
    try:
        for epoch in range(start_epoch, config['training']['epochs']):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
            logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            logger.info(f"{'='*80}")
            
            # Train
            train_loss, train_loss_items = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                sequence_length=config['dataset']['train']['seq_len'],
                scaler=scaler,
                epoch=epoch,
                logger=logger,
                max_grad_norm=config['training'].get('max_grad_norm', 10.0)
            )
            
            # Validate
            val_loss, val_loss_items = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
                sequence_length=config['dataset']['val']['seq_len'],
                epoch=epoch,
                logger=logger
            )
            
            # Step scheduler
            scheduler.step()
            
            # --- Save Checkpoints ---
            # Save the latest model
            latest_checkpoint_path = save_dir / "latest.pt"
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, val_loss,
                latest_checkpoint_path, config, logger
            )
            
            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = save_dir / "best.pt"
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, val_loss,
                    best_checkpoint_path, config, logger
                )
                logger.info(f"✓ New best model! Validation loss: {best_val_loss:.4f}")
            
            # Save periodic checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                periodic_path = save_dir / f"epoch_{epoch+1}.pt"
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, val_loss,
                    periodic_path, config, logger
                )
        
        logger.info("\n" + "="*80)
        logger.info("Training finished successfully!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Training completed by: Resnick28")
        logger.info(f"Training end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info("="*80)
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user!")
        logger.info("Saving interrupt checkpoint...")
        interrupt_path = save_dir / "interrupt.pt"
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, val_loss,
            interrupt_path, config, logger
        )
        logger.info(f"Interrupt checkpoint saved to {interrupt_path}")
        
    except Exception as e:
        logger.error(f"\nTraining failed with error: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOTemporalUNet on DSEC dataset')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., runs/train/exp1/latest.pt)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Update config path if provided
    if args.config != 'config.yaml':
        print(f"Using config: {args.config}")
    
    main(resume_from=args.resume)



# import yaml
# import cv2
# import torch
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from pathlib import Path

# # Make sure your DSECDataset class is in a file named dataset.py
# # in the same directory as this script.
# from dataset import DSECDataset

# # Class names corresponding to the class_id in your dataset
# CLASS_NAMES = {
#     0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'bus',
#     4: 'truck', 5: 'bicycle', 6: 'motorcycle', 7: 'train'
# }
# # Colors for the bounding boxes for each class
# CLASS_COLORS = plt.cm.get_cmap('hsv', len(CLASS_NAMES))

# def visualize_and_save_sample(image_tensor, labels_tensor, sample_index, output_dir):
#     """
#     Visualizes the last frame of a sequence and saves the plot to a file.

#     Args:
#         image_tensor (torch.Tensor): The tensor containing the sequence of images.
#         labels_tensor (torch.Tensor): The tensor containing the labels for the last frame.
#         sample_index (int): The index of the sample for the plot title and filename.
#         output_dir (Path): The directory where the visualization will be saved.
#     """
#     # --- 1. Prepare the image ---
#     # Convert the PyTorch tensor (C, H, W) to a NumPy array
#     last_image_np = image_tensor[-1].numpy()
    
#     # --- FIX: Transpose the array to the Matplotlib format (H, W, C) ---
#     last_image_np = last_image_np.transpose((1, 2, 0))

#     # Denormalize if the image tensor was scaled to [0.0, 1.0]
#     if last_image_np.dtype != np.uint8:
#         last_image_np = (last_image_np * 255).clip(0, 255).astype(np.uint8)

#     # --- 2. Create the plot ---
#     fig, ax = plt.subplots(1, figsize=(12, 8))
#     ax.imshow(last_image_np)
#     ax.axis('off')

#     # --- 3. Draw bounding boxes and labels ---
#     if labels_tensor.numel() == 0:
#         ax.set_title(f"Sample {sample_index}: Last Frame (No Detections)")
#     else:
#         ax.set_title(f"Sample {sample_index}: Last Frame with Bounding Boxes")
#         for box in labels_tensor:
#             class_id, x_center, y_center, width, height = box.numpy()
#             class_id = int(class_id)
            
#             # Convert center coordinates to top-left corner
#             x1 = x_center - width / 2
#             y1 = y_center - height / 2
            
#             class_name = CLASS_NAMES.get(class_id, f'Unknown: {class_id}')
#             color = CLASS_COLORS(class_id / len(CLASS_NAMES))
            
#             # Create a Rectangle patch
#             rect = patches.Rectangle(
#                 (x1, y1), width, height,
#                 linewidth=2, edgecolor=color, facecolor='none'
#             )
#             ax.add_patch(rect)
            
#             # Add the class label text
#             ax.text(
#                 x1, y1 - 5, class_name,
#                 color='white', backgroundcolor=color,
#                 fontsize=9, bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=1)
#             )

#     # --- 4. Save the figure to a file instead of showing it ---
#     output_filename = output_dir / f"sample_{sample_index}.png"
#     plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
#     print(f"Saved visualization to: {output_filename}")
    
#     # Close the figure to free up memory
#     plt.close(fig)

# if __name__ == '__main__':
#     # Use a non-interactive backend for SSH
#     plt.switch_backend('Agg')

#     # --- Load Configuration ---
#     config_path = Path('config.yaml')
#     if not config_path.exists():
#         raise FileNotFoundError("Error: config.yaml not found. Please create it and update the dataset paths.")

#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)

#     # --- Create output directory ---
#     output_dir = Path("./visualizations")
#     output_dir.mkdir(exist_ok=True)
#     print(f"Visualizations will be saved in: {output_dir.resolve()}")

#     print("Loading dataset...")
#     try:
#         train_dataset = DSECDataset(config=config, mode='train')
#     except Exception as e:
#         print(f"\n--- Could not initialize dataset ---")
#         print(f"Error: {e}")
#         print("Please check that the 'path' in your config.yaml points to the correct directory.")
#         exit()

#     # --- Visualize a few random samples ---
#     num_samples_to_show = 3
#     if len(train_dataset) < num_samples_to_show:
#         print(f"Warning: Dataset has fewer than {num_samples_to_show} samples. Showing all {len(train_dataset)} samples.")
#         num_samples_to_show = len(train_dataset)

#     if num_samples_to_show > 0:
#         print(f"Generating visualizations for {num_samples_to_show} random samples...")
#         random_indices = random.sample(range(len(train_dataset)), num_samples_to_show)

#         for i in random_indices:
#             image_tensor, labels_tensor = train_dataset[i]
#             visualize_and_save_sample(image_tensor, labels_tensor, sample_index=i, output_dir=output_dir)
#     else:
#         print("No samples found in the dataset to visualize.")

