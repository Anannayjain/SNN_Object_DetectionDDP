import time
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

# --- Imports from your project ---
from dataset import DSECDataset
from train import custom_collate_fn 

# --- Imports from Ultralytics ---
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou

# -------------------------------------------------------------------------
# 1. Stability Metric (tIoU)
# -------------------------------------------------------------------------
def compute_stability_score(pred_boxes_seq):
    """
    Calculates Temporal IoU (tIoU) across a sequence of frames.
    Input: List of tensors, where each tensor contains boxes for one frame [N, 6]
    Returns: Float (Average max IoU between frame t and t-1)
    """
    if len(pred_boxes_seq) < 2:
        return 0.0
        
    stability_scores = []
    
    for t in range(1, len(pred_boxes_seq)):
        curr_boxes = pred_boxes_seq[t]
        prev_boxes = pred_boxes_seq[t-1]
        
        # If no boxes detected in either frame, skip
        if curr_boxes is None or len(curr_boxes) == 0 or prev_boxes is None or len(prev_boxes) == 0:
            continue
            
        # Calculate IoU matrix between all current boxes and all previous boxes
        # curr: [N, 4], prev: [M, 4]
        ious = box_iou(curr_boxes[:, :4], prev_boxes[:, :4])
        
        if ious.numel() > 0:
            # For each object in current frame, find the best match (max IoU) in previous frame
            # High max_iou means the box didn't jump far (stable)
            max_ious, _ = ious.max(dim=1) 
            stability_scores.append(max_ious.mean().item())
        
    if not stability_scores:
        return 0.0 
        
    return np.mean(stability_scores)

# -------------------------------------------------------------------------
# 2. mAP Calculation Helper
# -------------------------------------------------------------------------
def compute_map_batch(pred_boxes, target_boxes, iou_thres=0.5):
    """
    Calculates Precision/Match rate for a single batch index (image).
    pred_boxes: [N, 6] (x1, y1, x2, y2, conf, cls) - Absolute Pixels
    target_boxes: [M, 5] (cls, x1, y1, x2, y2) - Absolute Pixels
    """
    if pred_boxes is None or len(pred_boxes) == 0:
        return 0.0
    if target_boxes is None or len(target_boxes) == 0:
        return 0.0

    # Sort predictions by confidence (descending)
    pred_boxes = pred_boxes[pred_boxes[:, 4].argsort(descending=True)]
    
    tp = 0
    matched_targets = set()
    
    for p_idx, pred in enumerate(pred_boxes):
        pred_cls = int(pred[5])
        pred_box = pred[:4]
        
        best_iou = 0
        best_t_idx = -1
        
        # Check against all targets of the same class
        for t_idx, target in enumerate(target_boxes):
            target_cls = int(target[0])
            
            if pred_cls != target_cls:
                continue
            
            # target[1:] contains coords
            iou = box_iou(pred_box.unsqueeze(0), target[1:].unsqueeze(0)).item()
            
            if iou > best_iou:
                best_iou = iou
                best_t_idx = t_idx
        
        # If match found above threshold and not already matched
        if best_iou >= iou_thres and best_t_idx not in matched_targets:
            tp += 1
            matched_targets.add(best_t_idx)
            
    # Simple mAP estimate: TP / Total_Targets (Recall-oriented accuracy)
    # For full COCO mAP, use torchmetrics, but this is sufficient for relative comparison
    return tp / (len(target_boxes) + 1e-6)

# -------------------------------------------------------------------------
# 3. Main Test Function
# -------------------------------------------------------------------------
def test_code(model, config, device):
    print("\n" + "="*50)
    print("   Starting DSEC Benchmark Test")
    print("="*50)

    # 1. Setup Output Directory
    save_dir = Path(config['training']['save_dir']) / "test_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Force Batch Size to 1
    # This is critical for temporal consistency and video grouping
    config['training']['batch_size'] = 1 
    
    # 3. Initialize Dataset
    print("Initializing Test Dataset...")
    # Ensure dataset.py is updated so 'test' mode loads labels!
    test_dataset = DSECDataset(config, mode="test") 
    
    # Debug Mode
    if config.get('debug_test', False):
        print("DEBUG MODE: Truncating to 100 samples.")
        subset_indices = list(range(min(100, len(test_dataset))))
        test_dataset = torch.utils.data.Subset(test_dataset, subset_indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1, 
        shuffle=False, 
        num_workers=config['training']['num_workers'],
        collate_fn=custom_collate_fn
    )

    # 4. Load Model Weights
    weights_path = Path(config['training']['weights_path'])
    if not weights_path.exists():
        # Fallback to 'best.pt' in save_dir if specified path doesn't exist
        fallback = Path(config['training']['save_dir']) / "best.pt"
        if fallback.exists():
            weights_path = fallback
    
    print(f"Loading weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)            
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 5. Metrics Storage
    video_stats = defaultdict(lambda: {"fps": [], "stability": [], "map": []})
    
    # Constants
    conf_thres_metric = 0.001  # Low threshold for mAP calculation
    iou_thres_nms = 0.6        # NMS IoU threshold
    
    print(f"Processing {len(test_dataset)} sequences...")
    
    # 6. Inference Loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Benchmarking"):
            
            # --- Unpack Batch (Now matches your corrected Dataset/Collate) ---
            # images: [1, Seq, C, H, W]
            # targets: [N, 6] -> [batch_idx, cls, cx, cy, w, h] (Normalized)
            # paths: List[str] (path to the last frame)
            images, targets, paths = batch 
            images = images.to(device)
            
            # Group metrics by Video Track (Parent folder name)
            video_id = str(Path(paths[0]).parent.name)
            
            # Get dimensions for Denormalization
            # images shape: [B, Seq, C, H, W]
            _, seq_len, _, h_img, w_img = images.shape
            
            # Temporal Inference
            seq_preds = [] 
            hidden_state = None
            start_time = time.time()
            
            for t in range(seq_len):
                frame = images[:, t, :, :, :]
                
                # Model Forward
                # Returns tuple (preds, hidden)
                out_tuple, hidden_state = model(frame, hidden_state)
                
                # NMS (Ultralytics)
                # Input: [B, 4+C, Anchors] (Raw Tensor)
                # Output: List[Tensor] -> [[x1,y1,x2,y2,conf,cls], ...] (Absolute Pixels)
                curr_preds = non_max_suppression(
                    out_tuple[0],
                    conf_thres=conf_thres_metric,
                    iou_thres=iou_thres_nms,
                    multi_label=True
                )
                
                # Store prediction for this frame (Batch index 0)
                # Clone to CPU to save GPU memory
                seq_preds.append(curr_preds[0].cpu())

            end_time = time.time()
            
            # --- Calculate Metrics for this Sequence ---
            
            # 1. FPS
            fps = seq_len / (end_time - start_time + 1e-6)
            video_stats[video_id]["fps"].append(fps)
            
            # 2. Stability (Jitter between frames)
            stability = compute_stability_score(seq_preds)
            video_stats[video_id]["stability"].append(stability)
            
            # 3. mAP (Evaluated on the LAST frame of the sequence)
            # We compare predictions of the last frame: seq_preds[-1]
            # against the targets for this sample.
            
            map_val = 0.0
            
            # Filter targets for batch index 0
            if targets is not None and len(targets) > 0:
                t_targets = targets[targets[:, 0] == 0]
                
                if len(t_targets) > 0:
                    # Denormalize targets to match prediction scale (Absolute Pixels)
                    # t_targets format: [batch_idx, cls, cx, cy, w, h]
                    
                    gt_cls = t_targets[:, 1].unsqueeze(1)
                    gt_bbox = t_targets[:, 2:].clone() # cx, cy, w, h
                    
                    # Multiply by tensor dimensions
                    gt_bbox[:, 0] *= w_img # cx
                    gt_bbox[:, 1] *= h_img # cy
                    gt_bbox[:, 2] *= w_img # w
                    gt_bbox[:, 3] *= h_img # h
                    
                    # Convert cxcywh -> xyxy
                    gt_xyxy = ops.xywh2xyxy(gt_bbox)
                    
                    # Combine: [cls, x1, y1, x2, y2]
                    formatted_targets = torch.cat([gt_cls, gt_xyxy], dim=1)
                    
                    # Compute mAP match
                    map_val = compute_map_batch(seq_preds[-1], formatted_targets)
            
            video_stats[video_id]["map"].append(map_val)

    # 7. Aggregate and Save Results
    print("\n" + "="*65)
    print(f"{'Video Track':<25} | {'mAP':<8} | {'FPS':<8} | {'Stability':<10}")
    print("-" * 65)
    
    global_map_acc = []
    results_json = {}
    
    for vid, stats in video_stats.items():
        avg_map = np.mean(stats["map"])
        avg_fps = np.mean(stats["fps"])
        avg_stab = np.mean(stats["stability"])
        
        global_map_acc.append(avg_map)
        
        print(f"{vid:<25} | {avg_map:<8.4f} | {avg_fps:<8.2f} | {avg_stab:<10.4f}")
        
        results_json[vid] = {
            "mAP": round(avg_map, 4),
            "FPS": round(avg_fps, 2),
            "Stability": round(avg_stab, 4)
        }
        
    global_map_final = np.mean(global_map_acc) if global_map_acc else 0.0
    
    print("-" * 65)
    print(f"Global mAP: {global_map_final:.4f}")
    print("=" * 65)
    
    # Save to JSON
    json_path = save_dir / "benchmark_metrics.json"
    final_output = {
        "global_map": round(global_map_final, 4),
        "video_breakdown": results_json
    }
    
    with open(json_path, "w") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"Detailed metrics saved to {json_path}")