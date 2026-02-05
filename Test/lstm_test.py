import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image  # Efficient image dimension reading

# --- Import from Ultralytics ---
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils import ops

def get_image_shape(path):
    """
    Reads image dimensions (W, H) without loading pixel data.
    """
    with Image.open(path) as img:
        return img.size

def run_inference(config, model, test_loader, output_dir, device, conf_thres=0.001):
    """
    Runs inference and saves bounding boxes to JSON files (one per sequence).
    """
    model.eval()
    sequence_length = config['dataset']['test']['seq_len']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Buffer to store results ---
    # Structure: { "sequence_name": [ { "frame": "img1.jpg", "boxes": [...] }, ... ] }
    results_buffer = {}

    print("Starting fast inference...")
    
    # Disable gradients
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Processing")
        
        for batch_idx, (image_tensor, last_frame_path_tuple) in enumerate(pbar):
            
            last_frame_path = last_frame_path_tuple[0]
            path_obj = Path(last_frame_path)
            
            sequence_name = path_obj.parents[3].name
            
            if sequence_name not in results_buffer:
                results_buffer[sequence_name] = []

            image_tensor = image_tensor.to(device)
            hidden_state = None
            
            # Temporal Loop
            for t in range(sequence_length):
                frame = image_tensor[:, t, :, :, :]
                preds, hidden_state = model(frame, hidden_state)

            # --- 3. NMS (On GPU) ---
            preds_post = non_max_suppression(
                preds[0],
                conf_thres=conf_thres,
                iou_thres=0.45,
                multi_label=True
            )
            
            preds_for_image = preds_post[0] # (N, 6)

            # --- 4. Fast Scaling (No Image Loading) ---
            boxes_to_save = []
            
            if preds_for_image is not None and len(preds_for_image) > 0:
                # Get Original Shape
                orig_w, orig_h = get_image_shape(last_frame_path)
                model_h, model_w = image_tensor.shape[-2:]

                # Scale boxes from Model Size -> Original Image Size
                scaled_boxes = ops.scale_boxes(
                    (model_h, model_w), 
                    preds_for_image[:, :4], 
                    (orig_h, orig_w)
                )
                
                # Combine scaled boxes with conf and class
                # final_preds: [x1, y1, x2, y2, conf, cls]
                final_preds = torch.cat((scaled_boxes, preds_for_image[:, 4:]), dim=1)

                # Convert to pure Python list for JSON serialization (Fast)
                # Round to 2 decimals to save disk space
                boxes_list = final_preds.cpu().numpy()
                
                
                for box in boxes_list:
                    x1, y1, x2, y2 = [float(x) for x in box[:4]]

                    # Subtract 15 from y coordinates
                    bbox = [
                        round(x1, 2), 
                        round(y1 - 15, 2), 
                        round(x2, 2), 
                        round(y2 - 15, 2)
                    ]
                    
                    boxes_to_save.append({
                        "class_id": int(box[5]),
                        "conf": round(float(box[4]), 4),
                        "bbox": bbox # [x1, y1, x2, y2]
                    })

            # --- Store in Buffer ---
            results_buffer[sequence_name].append({
                "frame_name": path_obj.name,
                "predictions": boxes_to_save
            })

    # --- Batch Save to Disk ---
    # Writing once per sequence avoids disk I/O bottlenecks
    print("Inference done. Saving results to JSON...")
    
    for seq_name, data in results_buffer.items():
        save_path = output_dir / f"{seq_name}.json"
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)
            
    print(f"All results saved to: {output_dir}")

# --- Usage ---
if __name__ == "__main__":
    # Ensure your config has the correct test paths
    # run_fast_inference(config, model, test_loader, "output_predictions", device)
    pass