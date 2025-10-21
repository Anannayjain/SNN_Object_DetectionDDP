
import torch
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from collections import defaultdict

from model import YOLOTemporalUNet
from dataset import DSECDataset
from train import custom_collate_fn, train_loop


def get_train_val_split(config, full_train_dataset, dataset_seq_len):    
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

if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset and DataLoaders ---
    full_train_dataset = DSECDataset(config, mode="train")
    dataset_seq_len = config['dataset']['train']['seq_len']
    train_dataset, val_dataset = get_train_val_split(config, full_train_dataset, dataset_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True, 
        collate_fn=custom_collate_fn
    )

    print(f"Total samples: {len(full_train_dataset)}. Train: {len(train_dataset)}. Val: {len(val_dataset)}.")

    # --- Model Initialization ---
    model = YOLOTemporalUNet(
        num_classes=config['model']['num_classes'],
        yolo_model_name=config['model']['yolo_model_name'],
        use_conv_lstm=config['model']['use_conv_lstm']
    ).to(device)
    model.train()

    if(config["mode"] == "train"):
         train_loop(model, train_loader, val_loader, config, device, save_dir)