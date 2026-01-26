import torch
import yaml
import cv2
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# --- Import from your project files ---
from model import YOLOTemporalUNet
from dataset import DSECDataset 

# --- Import from Ultralytics ---
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils import ops

import matplotlib.pyplot as plt

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
            x1, y1, x2, y2, cls_idx = box.cpu().numpy()
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
            x1, y1, x2, y2, conf, cls_idx = pred.cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_idx = int(cls_idx)
            
            # label = f"{class_names[cls_idx]} {conf:.2f}"
            color = colors[cls_idx]
            
            cv2.rectangle(image, (x1, y1-15), (x2, y2-15), color, 2)
            # (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            # cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), color, -1)        
            # cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
    return image

def run_visualization(config, model, vis_loader, output_dir, device, conf_thres=0.3): # Using 0.3 from your __main__
    """
    Runs inference on the vis set and saves visualized outputs.
    """
    sequence_length = config['dataset']['vis']['seq_len']
    
    # --- Setup Class Names and Colors ---
    cmap = plt.get_cmap('tab20')

    # Generate the list of RGB tuples (scaled to 0-255)
    num_classes = config['model']['num_classes']
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(num_classes)]
    class_names = [f"Class_{i}" for i in range(config['model']['num_classes'])]

    # --- Run Inference and Visualization Loop ---
    pbar = tqdm(vis_loader, desc="Visualizing")
    for batch_idx, (image_tensor, last_frame_path_tuple, labels_tensor) in enumerate(pbar):
        
        last_frame_path = last_frame_path_tuple[0]
        image_tensor = image_tensor.to(device)
        
        # --- Manual Model Forward ---
        hidden_state = None
        with torch.no_grad():
            for t in range(sequence_length):
                frame = image_tensor[:, t, :, :, :]
                # preds is a tuple: (concatenated_output, [feature_map_1, ...])
                preds, hidden_state = model(frame, hidden_state)

        preds_post = non_max_suppression(
            preds[0],
            conf_thres=conf_thres,
            iou_thres=0.45,
            multi_label=True # Set multi_label=True for standard NMS
        )
        # Get predictions for the single image in the batch
        # shape (N, 6) -> [x1, y1, x2, y2, conf, cls]
        preds_for_image = preds_post[0] 

        # --- Load Original Image for Drawing ---
        original_image = cv2.imread(last_frame_path)
        if original_image is None:
            print(f"Warning: Could not read image {last_frame_path}. Skipping.")
            continue
            
        # Only required if model input size differs from original size of image
        orig_h, orig_w = original_image.shape[:2]
        model_h, model_w = image_tensor.shape[-2:]

        if preds_for_image is not None and len(preds_for_image) > 0:
            scaled_preds_boxes = ops.scale_boxes(
                (model_h, model_w), 
                preds_for_image[:, :4], 
                (orig_h, orig_w)
            )
            scaled_preds = torch.cat((scaled_preds_boxes, preds_for_image[:, 4:]), dim=1)
        else:
            scaled_preds = None

        labels_tensor = labels_tensor[0]
        if labels_tensor is not None and len(labels_tensor) > 0:
            # labels_tensor is [class, cx, cy, w, h]            
            gt_boxes = labels_tensor.clone()

            gt_boxes[:, 1] *= orig_w  # cx
            gt_boxes[:, 2] *= orig_h  # cy
            gt_boxes[:, 3] *= orig_w  # w
            gt_boxes[:, 4] *= orig_h  # h
            
            x1 = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
            y1 = gt_boxes[:, 2] - gt_boxes[:, 4] / 2
            x2 = gt_boxes[:, 1] + gt_boxes[:, 3] / 2
            y2 = gt_boxes[:, 2] + gt_boxes[:, 4] / 2
            
            scaled_gt = torch.stack([x1, y1, x2, y2, gt_boxes[:, 0]], dim=1)

        image_with_boxes = draw_bboxes(original_image, scaled_preds, scaled_gt, class_names, colors)    

        path_obj = Path(last_frame_path)
        sequence_name = path_obj.parents[3].name
        seq_output_dir = output_dir / sequence_name
        seq_output_dir.mkdir(parents=True, exist_ok=True)        
        save_path = seq_output_dir / path_obj.name
        
        cv2.imwrite(str(save_path), image_with_boxes)

    print(f"\nVisualization complete. Results saved to {output_dir}")

if __name__ == "__main__":
    pass
    # run_visualization(config_path="config.yaml", conf_thres=0.3)
