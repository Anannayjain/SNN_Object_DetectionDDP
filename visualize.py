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


def draw_bboxes(image, predictions, class_names, colors):
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
    if predictions is None or len(predictions) == 0:
        return image

    for pred in predictions:
        x1, y1, x2, y2, conf, cls_idx = pred.cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_idx = int(cls_idx)
        
        label = f"{class_names[cls_idx]} {conf:.2f}"
        color = colors[cls_idx]
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        # (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), color, -1)        
        # cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return image

def run_visualization(config, model, test_loader, output_dir, device, conf_thres=0.3): # Using 0.3 from your __main__
    """
    Runs inference on the test set and saves visualized outputs.
    """
    sequence_length = config['dataset']['test']['seq_len']
    
    # --- Setup Class Names and Colors ---
    class_names = [f"Class_{i}" for i in range(config['model']['num_classes'])]
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(config['model']['num_classes'])]

    # --- Run Inference and Visualization Loop ---
    pbar = tqdm(test_loader, desc="Visualizing")
    for batch_idx, (image_tensor, last_frame_path_tuple) in enumerate(pbar):
        
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

        # --- Draw and Save ---
        image_with_boxes = draw_bboxes(original_image, scaled_preds, class_names, colors)
        
        save_path = output_dir / Path(last_frame_path).name
        cv2.imwrite(str(save_path), image_with_boxes)

    print(f"\nVisualization complete. Results saved to {output_dir}")

if __name__ == "__main__":
    pass
    # run_visualization(config_path="config.yaml", conf_thres=0.3)