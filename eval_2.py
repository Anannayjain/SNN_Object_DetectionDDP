import torch
import yaml
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from types import SimpleNamespace # To create a fake 'args' object

# --- Import from your project files ---
from model import YOLOTemporalUNet
from dataset import DSECDataset
from train import custom_collate_fn
from main import get_train_val_split # Assuming main.py contains this

# --- Import from Ultralytics ---
# We need the validator, metrics, and NMS
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils import ops

def evaluate_model(config_path="config.yaml"):
    """
    Runs evaluation on the validation set and prints mAP metrics.
    """
    # --- 1. Load Configuration ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = config["device"]
    save_dir = Path(config['training']['save_dir'])
    weights_path = save_dir / "best.pt" # Evaluate the best model
    sequence_length = config['dataset']['train']['seq_len']

    # --- 2. Load Model ---
    print(f"Loading model from {weights_path}...")
    model = YOLOTemporalUNet(
        num_classes=config['model']['num_classes'],
        yolo_model_name=config['model']['yolo_model_name'],
        use_conv_lstm=config['model']['use_conv_lstm']
    ).to(device)
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- 3. Load Validation Data ---
    # We use the 'train' dataset and split it, just like in training,
    # to get the *exact same* validation set.
    full_train_dataset = DSECDataset(config, mode="train")
    _, val_dataset = get_train_val_split(config, full_train_dataset, sequence_length)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=False, 
        collate_fn=custom_collate_fn
    )
    print(f"Loaded {len(val_dataset)} validation samples.")

    # --- 4. Setup Ultralytics Validator & Metrics ---
    # We need to create a dummy 'args' object that the validator expects
    validator_args = SimpleNamespace(
        save_dir=save_dir,
        device=device,
        model=config['model']['yolo_model_name'], # Used for default cfgs
        data='coco128.yaml', # Placeholder
        batch=config['training']['batch_size'],
        imgsz=640, # Assume 640, or get from config
        conf=0.001,  # Confidence threshold for mAP
        iou=0.6,     # IoU threshold for mAP
        max_det=300,
        split='val', # Important for the validator
        save_json=False,
        save_hybrid=False,
        project=None,
        name=None,
        # Add any other args your loss/validator might need
    )

    # Initialize the validator
    validator = DetectionValidator(args=validator_args)
    validator.device = device
    validator.data = {"nc": config['model']['num_classes']} # Tell it num_classes
    
    # Initialize the metrics computer
    validator.metrics = DetMetrics(save_dir=save_dir)
    validator.nt_per_class = torch.zeros(config['model']['num_classes'])

    # --- 5. Run Evaluation Loop ---
    pbar = tqdm(val_loader, desc="Evaluating")
    for batch_idx, (image_tensor, labels_tensor) in enumerate(pbar):
        image_tensor = image_tensor.to(device)
        labels_tensor = labels_tensor.to(device)
        
        # --- Manual Model Forward (for recurrent model) ---
        hidden_state = None
        with torch.no_grad():
            for t in range(sequence_length):
                frame = image_tensor[:, t, :, :, :]
                # 'preds' will be overwritten until the last frame
                preds, hidden_state = model(frame, hidden_state)
        
        # 'preds' is raw output from the last frame
        # 'labels_tensor' is GT for the last frame [batch_idx, cls, cx, cy, w, h]
        
        # --- Post-processing (NMS) ---
        preds_post = ops.non_max_suppression(
            preds,
            conf_thres=validator.args.conf,
            iou_thres=validator.args.iou
        )
        
        # --- Update Metrics ---
        # Prepare targets in the format DetMetrics expects
        targets = {}
        targets['batch_idx'] = labels_tensor[:, 0]
        targets['cls'] = labels_tensor[:, 1]
        targets['bboxes'] = labels_tensor[:, 2:]
        targets['img'] = image_tensor[:, -1, :, :, :] # Last frame
        
        # This is the core metrics update function
        validator.metrics.process(targets, preds_post)

    # --- 6. Print Results ---
    print("\nEvaluation finished. Calculating metrics...")
    stats = validator.metrics.results_dict()
    validator.metrics.print_results()
    
    return stats

if __name__ == "__main__":
    evaluate_model(config_path="config.yaml")