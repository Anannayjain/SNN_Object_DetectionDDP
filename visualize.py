import torch
import yaml
import cv2
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json

# --- Import from your project files ---
# from Models.lstm_model import YOLOTemporalUNet
from Dataset.lstm_dataset import DSECDataset 

# --- Import from Ultralytics ---
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils import ops

import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DSEC_CLASSES = {
    0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'bus', 
    4: 'truck', 5: 'bicycle', 6: 'motorcycle', 7: 'train'
}

def get_colors(num_classes=8):
    """Generate distinct colors for visualization."""
    cmap = plt.get_cmap('tab20')
    return [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(num_classes)]

def load_json_data(json_path, key_name):
    """
    Loads JSON and converts to Dict: { "000150.png": [objects...] }
    """
    if not Path(json_path).exists():
        print(f"Warning: File not found {json_path}")
        return {}
        
    with open(json_path, 'r') as f:
        data_list = json.load(f)
    
    data_dict = {}
    for item in data_list:
        data_dict[item['frame_name']] = item.get(key_name, [])
    return data_dict

def draw_bboxes(image, predictions, scaled_gt, class_names, colors):
    """
    Draws bounding boxes on an image.
    
    Args:
        image (np.ndarray): Image to draw on (BGR format from cv2.imread).
        predictions (torch.Tensor): Tensor of predictions from NMS,
                                   shape (N, 6) [x1, y1, x2, y2, conf, cls].
                                   These should be *scaled* to the image dimensions.
        class_names (list): List of class names.
        colors (list): List of (B, G, R) color tuples.
    """
    # if predictions is None or len(predictions) == 0:
    #     return image
    GT_COLOR = (0, 255, 0)
    if scaled_gt is not None and len(scaled_gt) > 0:
        for box in scaled_gt:
            x1, y1, x2, y2, cls_idx = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_idx = int(cls_idx)
            
            color = colors[cls_idx]
            # label = f"[GT] {class_names[cls_idx]}"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), GT_COLOR, 3)
            
            # # Label Background (Draw slightly above the box)
            # (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # # Label Text (White text on color background)
            # cv2.putText(image, label, (x1, y1 - 5), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if predictions is not None and len(predictions) > 0:
        for pred in predictions:
            x1, y1, x2, y2, conf, cls_idx = pred
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_idx = int(cls_idx)
            
            # label = f"{class_names[cls_idx]} {conf:.2f}"
            color = colors[cls_idx]
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            # cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), color, -1)        
            # cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
    return image

def visualize_sequence(seq_name, image_dir, pred_path, gt_path, output_dir):
    seq_out_dir = Path(output_dir) / seq_name
    seq_out_dir.mkdir(parents=True, exist_ok=True)

    preds_map = load_json_data(pred_path, key_name="predictions")
    gt_map = load_json_data(gt_path, key_name="annotations")
    
    all_frames = sorted(set(list(preds_map.keys()) + list(gt_map.keys())))
    
    if not all_frames:
        print(f"Skipping {seq_name}: No frames in JSONs.")
        return

    colors = get_colors()
    
    # Using tqdm leave=False to keep the main progress bar clean
    for frame_name in tqdm(all_frames, desc=f"Rendering {seq_name}", leave=False):
        img_path = Path(image_dir) / frame_name
        
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None: continue

        # GT Format: [x1, y1, x2, y2, cls]
        gt_objs = gt_map.get(frame_name, [])
        gt_list = [obj['bbox'] + [obj['class_id']] for obj in gt_objs]
        
        # Pred Format: [x1, y1, x2, y2, conf, cls]
        pred_objs = preds_map.get(frame_name, [])
        pred_list = [obj['bbox'] + [obj['conf'], obj['class_id']] for obj in pred_objs]

        image = draw_bboxes(image, pred_list, gt_list, DSEC_CLASSES, colors)
        cv2.imwrite(str(seq_out_dir / frame_name), image)

if __name__ == "__main__":
    # --- GLOBAL PATHS ---
    BASE_DSEC_DIR = Path("/home/ashutosh/pulkit/SNN_scratch_AJ/dsec_dataset/test")
    PRED_DIR = Path("runs/train/exp1/test_results") # Where your model outputs are
    GT_DIR = Path("ground_truth")             # Where your GT jsons are
    
    OUTPUT_DIR = Path("runs/train/exp1/visualizations")
    OUTPUT_DIR.mkdir(exist_ok=True)

    pred_files = sorted(list(PRED_DIR.glob("*.json")))
    print(f"Found {len(pred_files)} sequences to visualize.")

    for pred_file in tqdm(pred_files, desc="Total Progress"):
        seq_name = pred_file.stem  # e.g., "zurich_city_11_a"
        
        raw_image_dir = BASE_DSEC_DIR / seq_name / "images/left/distorted"
        gt_file = GT_DIR / f"{seq_name}.json"
        
        if not raw_image_dir.exists():
            print(f"Warning: Image folder not found for {seq_name}. Skipping.")
            continue
            
        # Run Visualization
        visualize_sequence(seq_name, raw_image_dir, pred_file, gt_file, OUTPUT_DIR)

    print(f"\nAll Done! Visualizations saved to: {OUTPUT_DIR.absolute()}")

def run_visualization():
    pass