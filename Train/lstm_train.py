import torch
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from ultralytics.utils.loss import v8DetectionLoss
from torch.utils.tensorboard import SummaryWriter
import os
# -----------------------------------------------------

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


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, sequence_length, scheduler=None, writer=None, epoch=0):
    # When you call loss.backward(), it will pinpoint the exact line of code in your forward pass that created a "bad" value (like a NaN or inf) that is causing the backward pass to fail.
    # torch.autograd.set_detect_anomaly(True)
    
    model.train()
    total_loss = 0.0
    total_loss_components = torch.zeros(3).to(device)
    pbar = tqdm(dataloader, desc="Training")
    
    
    for batch_idx, (image_tensor, labels_tensor) in enumerate(pbar):
        image_tensor = image_tensor.to(device)
        labels_tensor = labels_tensor.to(device)
        optimizer.zero_grad(set_to_none=True)
        hidden_state = None
        
        for t in range(sequence_length):
            frame = image_tensor[:, t, :, :, :]
            preds, hidden_state = model(frame, hidden_state)
        
        batch_dict = {
                'batch_idx': labels_tensor[:, 0],
                'cls': labels_tensor[:, 1],
                'bboxes': labels_tensor[:, 2:]
            }
        
        loss_components, loss_components_detached = loss_fn(preds, batch_dict)   
        scalar_loss = loss_components.sum()
        scalar_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)             
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += scalar_loss.item()   
        total_loss_components += loss_components_detached
        pbar.set_postfix(loss=f'{scalar_loss.item():.4f}')

        global_step = epoch * len(dataloader) + batch_idx
            
        # Log scalar loss
        writer.add_scalar('Loss/train_batch', scalar_loss.item(), global_step)
        writer.add_scalars('Train_Loss_Components_Batch', {
            'box_loss_batch': loss_components_detached[0].item(),
            'cls_loss_batch': loss_components_detached[1].item(),
            'dfl_loss_batch': loss_components_detached[2].item()
        }, global_step)
        print( {
            'box_loss_batch': loss_components_detached[0].item(),
            'cls_loss_batch': loss_components_detached[1].item(),
            'dfl_loss_batch': loss_components_detached[2].item()
        })
        # print(loss_components_detached.item())
        writer.add_scalar('LearningRate/batch', optimizer.param_groups[0]['lr'], global_step)

    avg_loss = total_loss / len(dataloader)
    avg_loss_comps = total_loss_components / len(dataloader)
    return avg_loss, avg_loss_comps

def validate_one_epoch(model, dataloader, loss_fn, device, sequence_length, writer=None, epoch=0):
    model.eval()
    total_loss = 0.0
    total_loss_components = torch.zeros(3).to(device)
    pbar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for batch_idx, (image_tensor, labels_tensor) in enumerate(pbar):
            image_tensor = image_tensor.to(device)
            labels_tensor = labels_tensor.to(device)
            
            hidden_state = None  
            for t in range(sequence_length):
                frame = image_tensor[:, t, :, :, :]
                preds, hidden_state = model(frame, hidden_state)            
            batch_dict = {
                    'batch_idx': labels_tensor[:, 0],
                    'cls': labels_tensor[:, 1],
                    'bboxes': labels_tensor[:, 2:]
                }
            _, loss_components_detached = loss_fn(preds, batch_dict)
            scalar_loss = loss_components_detached.sum()

            total_loss += scalar_loss.item() 
            total_loss_components += loss_components_detached             
                 
            pbar.set_postfix(loss=f'{scalar_loss.item():.4f}')

            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/val_batch', scalar_loss.item(), global_step)            
            writer.add_scalars('Val_Loss_Components_Batch', {
                'box_loss_batch': loss_components_detached[0].item(),
                'cls_loss_batch': loss_components_detached[1].item(),
                'dfl_loss_batch': loss_components_detached[2].item()
            }, global_step)

    avg_loss = total_loss / len(dataloader)
    avg_loss_comps = total_loss_components / len(dataloader)
    return avg_loss, avg_loss_comps

def train_loop(model, train_loader, val_loader, config, device, save_dir):

    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'runs'))
    
    # dummy_input = torch.randn(1, 3, 480, 640).to(device) # 1 frame
    # model(dummy_input, None)
    # writer.add_graph(model, dummy_input) # (frame, hidden_state)

    # --- Optimizer and Loss Function ---
    loss_fn = v8DetectionLoss(model)
    optimizer = optim.AdamW(
        model.parameters(), 
        # lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    total_steps = len(train_loader) * config['training']['epochs']

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'], # Your config LR is the *peak* LR
        total_steps=total_steps,
        pct_start=0.3, # 30% of total steps for increasing LR
        anneal_strategy='cos' # Use cosine annealing
    )
    best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        
        train_loss, train_loss_comps = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            loss_fn, 
            device, 
            config['dataset']['train']['seq_len'],
            scheduler=scheduler,
            writer=writer,
            epoch=epoch
        )
        print(f"Average Training Loss: {train_loss}")
        
        
        val_loss, val_loss_comps = validate_one_epoch(
            model, 
            val_loader, 
            loss_fn, 
            device, 
            config['dataset']['train']['seq_len'],
            writer=writer,
            epoch=epoch
        )
        print(f"Average Validation Loss: {val_loss}")
        
        # --- Save Checkpoints ---
        # Save the latest model
        latest_checkpoint_path = save_dir / "latest.pt"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_val_loss': best_val_loss  # Save the best loss *so far*
        }
        torch.save(checkpoint, latest_checkpoint_path)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log loss components using 'add_scalars' to group them
        writer.add_scalars('Train_Loss_Components', {
            'box_loss': train_loss_comps[0].item(),
            'cls_loss': train_loss_comps[1].item(),
            'dfl_loss': train_loss_comps[2].item()
        }, epoch)
        
        writer.add_scalars('Val_Loss_Components', {
            'box_loss': val_loss_comps[0].item(),
            'cls_loss': val_loss_comps[1].item(),
            'dfl_loss': val_loss_comps[2].item()
        }, epoch)

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = save_dir / "best.pt"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss
                # You can also save optimizer_state_dict here if you create it before the loop
            }
            torch.save(checkpoint, best_checkpoint_path)
            print(f"New best model saved to {best_checkpoint_path} with validation loss: {best_val_loss:.4f}")
        else:
            print(f"Saved latest model checkpoint to {latest_checkpoint_path}")


    print("\nTraining finished!")