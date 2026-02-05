import json
import torch
from pathlib import Path
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

def load_data(json_path, key_name):
    """
    Generic loader. 
    key_name: 'predictions' for pred file, 'annotations' for GT file.
    """
    with open(json_path, 'r') as f:
        data_list = json.load(f)
    
    # Convert list to dict for fast lookup: { "000150.png": [objects...] }
    data_dict = {}
    for item in data_list:
        data_dict[item['frame_name']] = item.get(key_name, [])
    return data_dict

def get_sequence_data(pred_path, gt_path):
    # 1. Load Data
    preds_map = load_data(pred_path, key_name="predictions")
    gt_map = load_data(gt_path, key_name="annotations")
    
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)
    preds_list = []
    target_list = []
    
    # 2. Iterate over all frames in GT
    all_frames = sorted(gt_map.keys())  
    print(f"Evaluating {len(all_frames)} frames...")
    
    for frame_name in tqdm(all_frames):
        # --- Prepare GT ---
        gt_objs = gt_map[frame_name]
        
        gt_boxes = [obj['bbox'] for obj in gt_objs]
        gt_labels = [obj['class_id'] for obj in gt_objs]
        
        target_dict = {
            "boxes": torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.empty(0, 4),
            "labels": torch.tensor(gt_labels, dtype=torch.int64) if gt_labels else torch.empty(0, dtype=torch.int64)
        }
        
        # --- Prepare Preds ---
        pred_objs = preds_map.get(frame_name, [])
        
        p_boxes = [obj['bbox'] for obj in pred_objs]
        p_scores = [obj['conf'] for obj in pred_objs]
        p_labels = [obj['class_id'] for obj in pred_objs]
        
        pred_dict = {
            "boxes": torch.tensor(p_boxes, dtype=torch.float32) if p_boxes else torch.empty(0, 4),
            "scores": torch.tensor(p_scores, dtype=torch.float32) if p_scores else torch.empty(0),
            "labels": torch.tensor(p_labels, dtype=torch.int64) if p_labels else torch.empty(0, dtype=torch.int64)
        }

        preds_list.append(pred_dict)
        target_list.append(target_dict)

    return preds_list, target_list

DSEC_CLASSES = {
    0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'bus', 
    4: 'truck', 5: 'bicycle', 6: 'motorcycle', 7: 'train'
}

def format_metrics_string(results, title):
    """Helper to format dictionary results into a readable string."""
    lines = []
    lines.append("="*40)
    lines.append(f"  {title}")
    lines.append("="*40)
    lines.append(f"mAP (50-95): {results['map']:.4f}")
    lines.append(f"mAP (50):    {results['map_50']:.4f}")
    lines.append(f"mAP (75):    {results['map_75']:.4f}")
    lines.append("-" * 40)
    lines.append("Per Class AP:")
    
    map_per_class = results['map_per_class']
    for cls_idx, score in enumerate(map_per_class):
        if score >= 0: 
            cls_name = DSEC_CLASSES.get(cls_idx, f"Class {cls_idx}")
            lines.append(f"  {cls_name:<12}: {score:.4f}")
            
    lines.append("\n")
    return "\n".join(lines)

if __name__ == "__main__":
    # --- PATHS ---
    GT_DIR = Path("ground_truth")
    PRED_DIR = Path("runs/train/exp1/test_results_vanilla")
    OUTPUT_FILE = Path("evaluation_report.txt")
    
    # --- METRICS ---
    global_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)
    local_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)

    pred_files = sorted(list(PRED_DIR.glob("*.json")))
    if not pred_files:
        print("No prediction files found.")
        exit()

    print(f"Evaluating {len(pred_files)} sequences. Writing to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w') as f_out:        
        for pred_file in tqdm(pred_files, desc="Evaluating"):
            seq_name = pred_file.stem
            gt_file = GT_DIR / f"{seq_name}.json"
            
            if not gt_file.exists():
                msg = f"Skipping {seq_name}: No GT found.\n"
                print(msg.strip())
                f_out.write(msg)
                continue
                
            preds_list, target_list = get_sequence_data(pred_file, gt_file)
            
            # --- LOCAL EVALUATION (Single Sequence) ---
            local_metric.update(preds_list, target_list)
            seq_results = local_metric.compute()
            
            report_str = format_metrics_string(seq_results, f"Sequence: {seq_name}")
            f_out.write(report_str)
            f_out.flush() # Ensure it writes to disk immediately
            
            # Reset local metric for next loop
            local_metric.reset()
            
            # --- GLOBAL EVALUATION (Accumulate) ---
            global_metric.update(preds_list, target_list)

        # 2. Compute Global Results (After loop ends)
        print("Computing Global Metrics...")
        global_results = global_metric.compute()
        
        global_report_str = format_metrics_string(global_results, "GLOBAL DATASET RESULTS")
        f_out.write(global_report_str)
        
    print(f"\nDone! Report saved to: {OUTPUT_FILE.absolute()}")
    print(global_report_str) # Print global stats to terminal at the end