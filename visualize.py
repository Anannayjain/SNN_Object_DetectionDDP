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
# Note: We don't need a custom collate_fn for test mode with batch_size=1

# --- Import from Ultralytics ---
# from ultralytics.utils.ops import non_max_suppression, scale_boxes
from ultralytics.utils.nms import non_max_suppression, xywh2xyxy
# from ultralytics.utils.ops import scale_boxes
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
        
        # Draw the rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw the label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        
        # Draw the label text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return image

# def run_visualization(config_path="config.yaml", conf_thres=0.01):
#     """
#     Runs inference on the test set and saves visualized outputs.
#     """
#     # --- 1. Load Configuration ---
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)

#     device = config["device"]
#     save_dir = Path(config['training']['save_dir'])
#     weights_path = save_dir / "best.pt"
#     sequence_length = config['dataset']['test']['seq_len']
#     num_classes = config['model']['num_classes']
    
#     # Create output directory
#     output_dir = save_dir / "visualizations"
#     output_dir.mkdir(parents=True, exist_ok=True)
#     print(f"Saving visualizations to {output_dir}")

#     # --- 2. Load Model ---
#     print(f"Loading model from {weights_path}...")
#     # model = YOLOTemporalUNet(
#     #     num_classes=num_classes,
#     #     yolo_model_name=config['model']['yolo_model_name'],
#     #     use_conv_lstm=config['model']['use_conv_lstm']
#     # ).to(device)
    
#     # --- AFTER (Fixed) ---
#     model = YOLOTemporalUNet(
#         num_classes=num_classes,
#         yolo_model_name=config['model']['yolo_model_name'],
#         use_conv_lstm=config['model']['use_conv_lstm'],
#         hyp=config['model']['hyp']  # <-- ADD THIS LINE
#     ).to(device)
    
#     model.load_state_dict(torch.load(weights_path, map_location=device))
#     model.eval()
#     print("Model loaded successfully.")
    
#     # --- 3. Setup Class Names and Colors ---
#     # Create placeholder class names
#     class_names = [f"Class_{i}" for i in range(num_classes)]
#     # Generate random colors for each class
#     np.random.seed(42)
#     colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)]

#     # --- 4. Load Test Data ---
#     # In 'test' mode, DSECDataset __getitem__ returns (image_tensor, last_frame_path)
#     test_dataset = DSECDataset(config, mode="test")
    
#     # Use batch_size=1 and default collate_fn
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=1, # Process one sequence at a time
#         shuffle=False,
#         num_workers=config['training']['num_workers']
#     )
#     print(f"Loaded {len(test_dataset)} test samples.")
    

#     # --- 5. Run Inference and Visualization Loop ---
#     pbar = tqdm(test_loader, desc="Visualizing")
#     for batch_idx, (image_tensor, last_frame_path_tuple) in enumerate(pbar):
        
#         # Unpack from batch_size=1
#         # image_tensor shape: (1, S, C, H, W)
#         # last_frame_path_tuple: ('/path/to/image.png',)
#         last_frame_path = last_frame_path_tuple[0]
        
#         image_tensor = image_tensor.to(device)
        
#         # --- Manual Model Forward ---
#         hidden_state = None
#         with torch.no_grad():
#             for t in range(sequence_length):
#                 frame = image_tensor[:, t, :, :, :]
#                 preds, hidden_state = model(frame, hidden_state)
#             # print(preds[0].shape, preds, preds[0].shape)
#         # --- Post-processing (NMS) ---
#         # preds is raw output from the last frame for batch_size=1
#         # preds_post = non_max_suppression(
#         #     preds,
#         #     conf_thres=conf_thres, # Use a reasonable confidence
#         #     iou_thres=0.01
#         # )
#         # preds (1, C+4, 8400) -> (8400, C+4)
#         preds_single = preds[0][0].permute(1, 0)
#         conf, cls_idx = preds_single[:, 4:].max(1, keepdim=True)
#         detections = torch.cat((preds_single[:, :4], conf, cls_idx.float()), 1)
#         detections = detections[detections[:, 4] >= 0]
#         preds_for_image = torch.empty(0, 6, device=device)
#         if detections.shape[0] > 0:
#             detections[:, :4] = xywh2xyxy(detections[:, :4])
#             preds_for_image = detections
        
#         # Get predictions for the single image in the batch
#         # shape (N, 6) -> [x1, y1, x2, y2, conf, cls]
#         # preds_for_image = preds_post[0] 
        
#         # --- Load Original Image for Drawing ---
#         original_image = cv2.imread(last_frame_path)
#         if original_image is None:
#             print(f"Warning: Could not read image {last_frame_path}. Skipping.")
#             continue
            
#         orig_h, orig_w = original_image.shape[:2]
#         model_h, model_w = image_tensor.shape[-2:] # Get model input size (e.g., 640, 640)
        
#         # --- CRITICAL: Scale boxes ---
#         # Predictions are relative to the model's input size (model_h, model_w)
#         # We must scale them to the original image size (orig_h, orig_w)
#         if preds_for_image is not None and len(preds_for_image) > 0:
#             scaled_preds = ops.scale_boxes(
#                 (model_h, model_w), 
#                 preds_for_image[:, :4], 
#                 (orig_h, orig_w)
#             )
#             # Put scaled boxes back with conf and cls
#             scaled_preds = torch.cat((scaled_preds, preds_for_image[:, 4:]), dim=1)
#         else:
#             scaled_preds = None

#         # --- Draw and Save ---
#         image_with_boxes = draw_bboxes(original_image, scaled_preds, class_names, colors)
        
#         save_path = output_dir / Path(last_frame_path).name
#         cv2.imwrite(str(save_path), image_with_boxes)

#     print(f"\nVisualization complete. Results saved to {output_dir}")

def run_visualization(config_path="config.yaml", conf_thres=0.3): # Using 0.3 from your __main__
    """
    Runs inference on the test set and saves visualized outputs.
    """
    # --- 1. Load Configuration ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = config["device"]
    save_dir = Path(config['training']['save_dir'])
    weights_path = save_dir / "best.pt"
    sequence_length = config['dataset']['test']['seq_len']
    num_classes = config['model']['num_classes']
    
    # Create output directory
    output_dir = save_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to {output_dir}")

    # --- 2. Load Model ---
    print(f"Loading model from {weights_path}...")
    model = YOLOTemporalUNet(
        num_classes=num_classes,
        yolo_model_name=config['model']['yolo_model_name'],
        use_conv_lstm=config['model']['use_conv_lstm'],
        hyp=config['model']['hyp']  # <-- FIX 1: Added hyp
    ).to(device)
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    
    # --- 3. Setup Class Names and Colors ---
    class_names = [f"Class_{i}" for i in range(num_classes)]
    np.random.seed(42)
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)]

    # --- 4. Load Test Data ---
    test_dataset = DSECDataset(config, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    print(f"Loaded {len(test_dataset)} test samples.")
    

    # --- 5. Run Inference and Visualization Loop ---
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
        
        # --- FIX 2: Replaced manual NMS ---
        # preds[0] is the concatenated tensor (B, C, N_Anchors) e.g. (1, 72, 8400)
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
            
        orig_h, orig_w = original_image.shape[:2]
        # Get model input size (e.g., 480, 640)
        model_h, model_w = image_tensor.shape[-2:]
        
        # --- CRITICAL: Scale boxes ---
        if preds_for_image is not None and len(preds_for_image) > 0:
            # Scale boxes from model_size (model_h, model_w) to original_size (orig_h, orig_w)
            scaled_preds_boxes = ops.scale_boxes(
                (model_h, model_w), 
                preds_for_image[:, :4], 
                (orig_h, orig_w)
            )
            # Put scaled boxes back with conf and cls
            scaled_preds = torch.cat((scaled_preds_boxes, preds_for_image[:, 4:]), dim=1)
        else:
            scaled_preds = None

        # --- Draw and Save ---
        image_with_boxes = draw_bboxes(original_image, scaled_preds, class_names, colors)
        
        save_path = output_dir / Path(last_frame_path).name
        cv2.imwrite(str(save_path), image_with_boxes)

    print(f"\nVisualization complete. Results saved to {output_dir}")

if __name__ == "__main__":
    run_visualization(config_path="config.yaml", conf_thres=0.3)