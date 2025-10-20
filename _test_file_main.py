import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import os

# Assuming your model and dataset are in separate files
from model import YOLOTemporalUNet
from dataset import DSECDataset

class YOLOLoss(nn.Module):
    """
    Custom YOLO-style loss function for multi-scale detection.
    Combines classification, objectness, and bounding box regression losses.
    """
    def __init__(self, num_classes=2, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, predictions, targets, feature_sizes):
        """
        Args:
            predictions: Tuple of (det_p3, det_p4, det_p5) each of shape (B, num_classes+5, H, W)
            targets: List of target tensors for each image in batch, shape (num_boxes, 5)
                     Format: [class_id, x_center, y_center, width, height] (normalized 0-1)
            feature_sizes: List of (H, W) tuples for each prediction scale
        """
        total_loss = 0.0
        batch_size = predictions[0].shape[0]
        device = predictions[0].device
        
        loss_dict = {
            'bbox_loss': 0.0,
            'obj_loss': 0.0,
            'cls_loss': 0.0,
            'total_loss': 0.0
        }
        
        for scale_idx, (pred, (feat_h, feat_w)) in enumerate(zip(predictions, feature_sizes)):
            # Reshape: (B, num_classes+5, H, W) -> (B, H, W, num_classes+5)
            pred = pred.permute(0, 2, 3, 1).contiguous()
            
            # Split predictions
            pred_boxes = pred[..., :4]  # (B, H, W, 4)
            pred_obj = pred[..., 4:5]   # (B, H, W, 1)
            pred_cls = pred[..., 5:]    # (B, H, W, num_classes)
            
            # Build target tensors
            for batch_idx in range(batch_size):
                if len(targets) <= batch_idx or targets[batch_idx].shape[0] == 0:
                    # No objects - only penalize objectness
                    obj_mask = torch.zeros_like(pred_obj[batch_idx])
                    loss_dict['obj_loss'] += self.lambda_noobj * self.bce_loss(
                        pred_obj[batch_idx], obj_mask
                    ).mean()
                    continue
                
                target = targets[batch_idx]
                
                # Create target maps
                obj_mask = torch.zeros((feat_h, feat_w, 1), device=device)
                noobj_mask = torch.ones((feat_h, feat_w, 1), device=device)
                target_boxes = torch.zeros((feat_h, feat_w, 4), device=device)
                target_cls = torch.zeros((feat_h, feat_w, self.num_classes), device=device)
                
                # Assign targets to grid cells
                for box in target:
                    cls_id = int(box[0].item())
                    x_center, y_center, width, height = box[1:5]
                    
                    # Convert to grid coordinates
                    grid_x = int(x_center * feat_w)
                    grid_y = int(y_center * feat_h)
                    
                    # Clamp to valid range
                    grid_x = min(max(grid_x, 0), feat_w - 1)
                    grid_y = min(max(grid_y, 0), feat_h - 1)
                    
                    # Assign target
                    obj_mask[grid_y, grid_x, 0] = 1.0
                    noobj_mask[grid_y, grid_x, 0] = 0.0
                    target_boxes[grid_y, grid_x] = torch.tensor(
                        [x_center, y_center, width, height], device=device
                    )
                    if cls_id < self.num_classes:
                        target_cls[grid_y, grid_x, cls_id] = 1.0
                
                # Calculate losses for this batch item
                obj_mask_bool = obj_mask.squeeze(-1) > 0.5
                
                # Bbox loss (only for cells with objects)
                if obj_mask_bool.any():
                    pred_boxes_obj = pred_boxes[batch_idx][obj_mask_bool]
                    target_boxes_obj = target_boxes[obj_mask_bool]
                    
                    # Use MSE for bbox coordinates and size
                    bbox_loss = self.mse_loss(pred_boxes_obj, target_boxes_obj).sum()
                    loss_dict['bbox_loss'] += self.lambda_coord * bbox_loss
                
                # Objectness loss
                obj_loss = self.bce_loss(pred_obj[batch_idx], obj_mask).mean()
                noobj_loss = self.lambda_noobj * (
                    self.bce_loss(pred_obj[batch_idx], obj_mask) * noobj_mask
                ).mean()
                loss_dict['obj_loss'] += obj_loss + noobj_loss
                
                # Classification loss (only for cells with objects)
                if obj_mask_bool.any():
                    pred_cls_obj = pred_cls[batch_idx][obj_mask_bool]
                    target_cls_obj = target_cls[obj_mask_bool]
                    cls_loss = self.bce_loss(pred_cls_obj, target_cls_obj).mean()
                    loss_dict['cls_loss'] += cls_loss
        
        # Average losses
        num_scales = len(predictions)
        for key in loss_dict:
            loss_dict[key] /= (batch_size * num_scales)
        
        loss_dict['total_loss'] = (
            loss_dict['bbox_loss'] + 
            loss_dict['obj_loss'] + 
            loss_dict['cls_loss']
        )
        
        return loss_dict


class Trainer:
    def __init__(self, config_path):
        """Initialize trainer with configuration file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.setup_device()
        self.setup_model()
        self.setup_data()
        self.setup_training()
        
    def setup_logging(self):
        """Setup logging and save directories."""
        self.save_dir = Path(self.config['training']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        log_file = self.save_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging to {log_file}")
        
    def setup_device(self):
        """Setup device (GPU/CPU) and seed."""
        torch.manual_seed(self.config['training']['seed'])
        np.random.seed(self.config['training']['seed'])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA version: {torch.version.cuda}")
            
    def setup_model(self):
        """Initialize model and loss function."""
        model_cfg = self.config['model']
        self.model = YOLOTemporalUNet(
            num_classes=model_cfg['num_classes'],
            yolo_model_name=model_cfg['yolo_model_name'],
            feature_channels=model_cfg['feature_channels'],
            use_conv_lstm=model_cfg['use_conv_lstm']
        ).to(self.device)
        
        self.criterion = YOLOLoss(
            num_classes=model_cfg['num_classes'],
            lambda_coord=5.0,
            lambda_noobj=0.5
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
    def setup_data(self):
        """Setup data loaders."""
        train_cfg = self.config['training']
        
        self.train_dataset = DSECDataset(self.config, mode='train')
        self.val_dataset = DSECDataset(self.config, mode='val')
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=True,
            num_workers=train_cfg['num_workers'],
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=False,
            num_workers=train_cfg['num_workers'],
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_dataset)}")
        
    def collate_fn(self, batch):
        """Custom collate function to handle variable number of boxes."""
        images = []
        labels = []
        
        for img, label in batch:
            images.append(img)
            labels.append(label)
        
        images = torch.stack(images, dim=0)
        return images, labels
        
    def setup_training(self):
        """Setup optimizer, scheduler, and other training components."""
        train_cfg = self.config['training']
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg['epochs'],
            eta_min=train_cfg['learning_rate'] * 0.01
        )
        
        self.scaler = GradScaler()
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {'bbox_loss': 0.0, 'obj_loss': 0.0, 'cls_loss': 0.0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # images: (B, seq_len, C, H, W)
            batch_size, seq_len, C, H, W = images.shape
            images = images.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Process sequence and accumulate loss
            hidden_state = None
            batch_loss = 0.0
            batch_loss_components = {'bbox_loss': 0.0, 'obj_loss': 0.0, 'cls_loss': 0.0}
            
            for t in range(seq_len):
                frame = images[:, t]  # (B, C, H, W)
                
                with autocast():
                    predictions, hidden_state = self.model(frame, hidden_state)
                    
                    # Only compute loss for the last frame
                    if t == seq_len - 1:
                        # Get feature sizes
                        feature_sizes = [
                            (pred.shape[2], pred.shape[3]) for pred in predictions
                        ]
                        
                        loss_dict = self.criterion(predictions, labels, feature_sizes)
                        batch_loss = loss_dict['total_loss']
                        
                        for key in batch_loss_components:
                            batch_loss_components[key] += loss_dict[key].item()
                
                # Detach hidden state to prevent backprop through time
                if hidden_state is not None:
                    if isinstance(hidden_state, tuple):
                        hidden_state = tuple(h.detach() for h in hidden_state)
                    else:
                        hidden_state = hidden_state.detach()
            
            # Backward pass
            self.scaler.scale(batch_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            total_loss += batch_loss.item()
            for key in loss_components:
                loss_components[key] += batch_loss_components[key]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': batch_loss.item(),
                'bbox': batch_loss_components['bbox_loss'],
                'obj': batch_loss_components['obj_loss'],
                'cls': batch_loss_components['cls_loss']
            })
        
        # Calculate averages
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        loss_components = {'bbox_loss': 0.0, 'obj_loss': 0.0, 'cls_loss': 0.0}
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for images, labels in pbar:
            batch_size, seq_len, C, H, W = images.shape
            images = images.to(self.device)
            
            # Process sequence
            hidden_state = None
            
            for t in range(seq_len):
                frame = images[:, t]
                
                with autocast():
                    predictions, hidden_state = self.model(frame, hidden_state)
                    
                    # Only compute loss for the last frame
                    if t == seq_len - 1:
                        feature_sizes = [
                            (pred.shape[2], pred.shape[3]) for pred in predictions
                        ]
                        
                        loss_dict = self.criterion(predictions, labels, feature_sizes)
                        total_loss += loss_dict['total_loss'].item()
                        
                        for key in loss_components:
                            loss_components[key] += loss_dict[key].item()
                
                if hidden_state is not None:
                    if isinstance(hidden_state, tuple):
                        hidden_state = tuple(h.detach() for h in hidden_state)
                    else:
                        hidden_state = hidden_state.detach()
            
            pbar.set_postfix({'loss': loss_dict['total_loss'].item()})
        
        # Calculate averages
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save last checkpoint
        last_path = self.save_dir / 'last.pt'
        torch.save(checkpoint, last_path)
        self.logger.info(f"Saved checkpoint: {last_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['val_loss']
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            # Train
            train_loss, train_components = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_components = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f} "
                f"(bbox: {train_components['bbox_loss']:.4f}, "
                f"obj: {train_components['obj_loss']:.4f}, "
                f"cls: {train_components['cls_loss']:.4f}) - "
                f"Val Loss: {val_loss:.4f} "
                f"(bbox: {val_components['bbox_loss']:.4f}, "
                f"obj: {val_components['obj_loss']:.4f}, "
                f"cls: {val_components['cls_loss']:.4f}) - "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOTemporalUNet on DSEC dataset')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Trainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()





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

