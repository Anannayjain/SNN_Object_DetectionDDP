
import torch
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from collections import defaultdict

from model import YOLOTemporalUNet
from dataset import DSECDataset
from visualize import run_visualization
from train import custom_collate_fn, train_loop
from weight_initialization import initialize_model

def get_train_val_split(config, full_train_dataset):    
    seq_groups = defaultdict(list)
    for idx, (img_dir, _, _) in enumerate(full_train_dataset.samples):
        seq_groups[str(img_dir)].append(idx)
    train_seqs, val_seqs = train_test_split(list(seq_groups), test_size=0.2, random_state=42)
    train_seqs, val_seqs = set(train_seqs), set(val_seqs)  # O(1) lookup
    train_indices, val_indices = [], []
    for seq, indices in seq_groups.items():
        (train_indices if seq in train_seqs else val_indices).extend(indices)
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)
    return train_dataset, val_dataset

def apply_train_debug_mode(config, train_dataset, val_dataset):
    is_debug = config.get('debug_train', False)
    if not is_debug:
        return train_dataset, val_dataset

    print("DEBUG MODE: Using a smaller subset for quick iterations.")    
    original_dataset = train_dataset.dataset 

    debug_train_indices = train_dataset.indices[:100] # First 100 train samples
    debug_val_indices = val_dataset.indices[:20]     # First 20 val samples

    debug_train_dataset = Subset(original_dataset, debug_train_indices)
    debug_val_dataset = Subset(original_dataset, debug_val_indices)
    
    print(f"DEBUG: Truncated to {len(debug_train_dataset)} train samples and {len(debug_val_dataset)} val samples.")
    
    return debug_train_dataset, debug_val_dataset

def train_code(model, config, device):
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset and DataLoaders ---
    full_train_dataset = DSECDataset(config, mode="train")

    train_dataset, val_dataset = get_train_val_split(config, full_train_dataset)
    train_dataset, val_dataset = apply_train_debug_mode(config, train_dataset, val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=False,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=False, 
        collate_fn=custom_collate_fn
    )

    print(f"Total samples: {len(full_train_dataset)}. Train: {len(train_dataset)}. Val: {len(val_dataset)}.")

    train_loop(model, train_loader, val_loader, config, device, save_dir)

def apply_test_debug_mode(config, test_dataset):
    is_debug = config.get('debug_test', False)
    if not is_debug:
        return test_dataset

    print("DEBUG MODE: Using a smaller subset for quick iterations.")
    num_debug_samples = min(100, len(test_dataset))
    debug_test_indices = list(range(num_debug_samples))
    debug_test_dataset = Subset(test_dataset, debug_test_indices)
    print(f"DEBUG: Truncated to {len(debug_test_dataset)} test samples.")

    return debug_test_dataset


def visualize_code(model, config, device):
    save_dir = Path(config['training']['save_dir'])
    weights_path = save_dir / "best.pt"
    
    # Create output directory
    output_dir = save_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to {output_dir}")
    
    checkpoint = torch.load(weights_path, map_location=device)            
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model with val loss {checkpoint.get('best_val_loss', float('inf'))} loaded successfully for visualization.")

    test_dataset = DSECDataset(config, mode="test")
    test_dataset = apply_test_debug_mode(config, test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=config['training']['num_workers']
    )

    print(f"Loaded {len(test_dataset)} test samples.")

    run_visualization(config, model, test_loader, output_dir, device)

if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)

    device = config["device"]

    # --- Model Initialization ---
    model = YOLOTemporalUNet(
        num_classes=config['model']['num_classes'],
        yolo_model_name=config['model']['yolo_model_name'],
        use_conv_lstm=config['model']['use_conv_lstm'],
        hyp=config['model']['hyp']
    ).to(device)
    
    initial_best_loss = float('inf') 

    if config['training']["resume_training"]:
        weights_path = Path(config['training']["weights_path"])
        if weights_path.exists():
            print(f"Resuming training: Loading from {weights_path}")
            checkpoint = torch.load(weights_path, map_location=device)            
            model.load_state_dict(checkpoint['model_state_dict'])
            initial_best_loss = checkpoint.get('best_val_loss', float('inf'))             
            print(f"Successfully loaded model and found previous best_val_loss: {initial_best_loss}")
            
        else:
            print(f"WARNING: 'resume_training' is True but weights_path '{weights_path}' not found.")
            print("Initializing model from scratch...")
            initialize_model(model)
    else:
        # Default behavior: initialize new model
        print("Initializing new model from scratch...")
        initialize_model(model)
    
    if(config["mode"] == "train"):
        train_code(model, config, device)
    elif(config["mode"] == "visualize"): 
        visualize_code(model, config, device)
    elif(config["mode"] == "test"):
        pass  # Testing code to be implemented
        