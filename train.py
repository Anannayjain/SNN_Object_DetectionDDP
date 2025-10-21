import torch
from tqdm import tqdm
import torch.optim as optim

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.models.yolo.detect.train import DetectionTrainer

def custom_collate_fn(batch):
    """
    Custom collate function to handle samples with a variable number of labels.
    """
    # batch is a list of tuples: [(img_seq_1, labels_1), (img_seq_2, labels_2), ...]
    # img_seq_1 shape: (S, C, H, W)
    # labels_1 shape: (N, 5)  (where N is variable)
    
    img_sequences = []
    labels_list = []
    
    for i, (img_seq, labels) in enumerate(batch):
        img_sequences.append(img_seq)        
        # Check if there are any labels
        if labels.shape[0] > 0:
            # Create a batch index tensor, shape (N, 1)
            # This 'i' is the batch_index for this sample
            batch_index = torch.full((labels.shape[0], 1), i, dtype=labels.dtype, device=labels.device)            
            # Prepend batch index: (N, 5) -> (N, 6)
            # [class, x, y, w, h] -> [batch_idx, class, x, y, w, h]
            labels_with_index = torch.cat([batch_index, labels], dim=1)
            labels_list.append(labels_with_index)
    # Stack all image sequences: list of (S, C, H, W) -> (B, S, C, H, W)
    images_batch = torch.stack(img_sequences, 0)
    
    # Concatenate all labels into one big tensor (Total_Objects, 6)
    if labels_list:
        labels_batch = torch.cat(labels_list, 0)
    else:
        # Handle case where no labels are in the entire batch
        # We still need 6 columns to match the expected format
        # Use float() to match image_tensor type, though it will be moved to device later
        labels_batch = torch.empty(0, 6, dtype=torch.float32) 
        
    return images_batch, labels_batch
# -----------------------------------------------------


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, sequence_length):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (image_tensor, labels_tensor) in enumerate(pbar):
        image_tensor = image_tensor.to(device)
        labels_tensor = labels_tensor.to(device)
        optimizer.zero_grad()
        hidden_state = None
        
        for t in range(sequence_length):
            frame = image_tensor[:, t, :, :, :]
            preds, hidden_state = model(frame, hidden_state)
            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach()) # truncate gradients
            
            batch_dict = {
                    'batch_idx': labels_tensor[:, 0],
                    'cls': labels_tensor[:, 1],
                    'bboxes': labels_tensor[:, 2:]
                }        
            # loss_components is a tensor of size [3] (box, cls, dfl)
            loss_components, _ = loss_fn(preds, batch_dict)        
            # Sum the components to get the final scalar loss
            scalar_loss = loss_components.sum()
            # Backpropagation
            scalar_loss.backward()               
            optimizer.step()        
            total_loss += scalar_loss.item() # Use the scalar_loss        
        # Update progress bar description
        pbar.set_postfix(loss=f'{scalar_loss.item():.4f}') # Use the scalar_loss
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, loss_fn, device, sequence_length):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for batch_idx, (image_tensor, labels_tensor) in enumerate(pbar):
            # ... (image/label moving and sequence loop are all correct) ...            
            for t in range(sequence_length):
                frame = image_tensor[:, t, :, :, :]
                preds, hidden_state = model(frame, hidden_state)            
            batch_dict = {
                    'batch_idx': labels_tensor[:, 0],
                    'cls': labels_tensor[:, 1],
                    'bboxes': labels_tensor[:, 2:]
                }
            loss_components, _ = loss_fn(preds, batch_dict)
            scalar_loss = loss_components.sum() # Sum components            
            total_loss += scalar_loss.item() # Use scalar_loss            
            pbar.set_postfix(loss=f'{scalar_loss.item():.4f}') # Use scalar_loss

    return total_loss / len(dataloader)

def train_loop(model, train_loader, val_loader, config, device, save_dir):
    # --- Loss Function, Optimizer ---
    overrides_cfg = {
        'model': config["model"]['yolo_model_name'],  # A base model for the trainer to load default cfgs
        'data': 'coco128.yaml',   # Placeholder, not actually used for data loading
        'epochs': config['training']['epochs'],
        'imgsz': 640,
        # === Key arguments for the loss function ===
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    }

    # Use a dummy trainer to setup the loss function correctly
    trainer = DetectionTrainer(overrides=overrides_cfg)
    trainer.model = model
    trainer.args.nc = config['model']['num_classes']
    model.args = trainer.args # The loss function reads properties from model.args

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
            config['dataset']['train']['seq_len']
        )
        print(f"Average Validation Loss: {val_loss:.4f}")
        
        # --- Save Checkpoints ---
        # Save the latest model
        latest_checkpoint_path = save_dir / "latest.pt"
        torch.save(model.state_dict(), latest_checkpoint_path)

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = save_dir / "best.pt"
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"New best model saved to {best_checkpoint_path} with validation loss: {best_val_loss:.4f}")
        else:
            print(f"Saved latest model checkpoint to {latest_checkpoint_path}")


    print("\nTraining finished!")