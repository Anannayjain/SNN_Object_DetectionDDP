# test_vanilla.py
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.utils import ops

# Reuse your existing dataset
from Dataset.lstm_dataset import DSECDataset

def get_image_shape(path):
    with Image.open(path) as img:
        return img.size

def run_vanilla_test(config, device):
    """
    Standalone function to test Vanilla YOLO on the DSEC dataset.
    """
    save_dir = Path(config['training']['save_dir'])
    output_dir = save_dir / "test_results_vanilla"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Model (Ultralytics handles loading internally)
    yolo_name = config['model']['yolo_model_name']
    print(f"--- VANILLA MODE ---")
    print(f"Loading backbone: {yolo_name}")
    model = YOLO(yolo_name) 

    CLASS_MAPPING = {
        0: 0,  # COCO Person     -> DSEC Pedestrian (Index 0)
        2: 2,  # COCO Car        -> DSEC Car        (Index 2)
        5: 3,  # COCO Bus        -> DSEC Bus        (Index 3)
        7: 4,  # COCO Truck      -> DSEC Truck      (Index 4)
        1: 5,  # COCO Bicycle    -> DSEC Bicycle    (Index 5)
        3: 6   # COCO Motorcycle -> DSEC Motorcycle (Index 6)
    }

    # print("Using Class Mapping (COCO -> DSEC):", CLASS_MAPPING)

    # 2. Setup Data Loader
    test_dataset = DSECDataset(config, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=config['training']['num_workers']
    )

    print(f"Processing {len(test_dataset)} sequences...")
    results_buffer = {}

    # 3. Inference Loop
    for batch_idx, (image_tensor, last_frame_path_tuple) in enumerate(tqdm(test_loader)):
        
        last_frame_path = last_frame_path_tuple[0]
        path_obj = Path(last_frame_path)
        sequence_name = path_obj.parents[3].name
        
        if sequence_name not in results_buffer:
            results_buffer[sequence_name] = []

        # Extract only the last frame
        # Shape: (1, 5, 3, H, W) -> (1, 3, H, W)
        last_frame_tensor = image_tensor[:, -1, :, :, :].to(device)

        # Inference
        results = model.predict(last_frame_tensor, verbose=False, device=device, conf=0.001, iou=0.45)
        
        # Post-Processing
        result = results[0]
        preds = result.boxes.data
        boxes_to_save = []

        if len(preds) > 0:
            orig_w, orig_h = get_image_shape(last_frame_path)
            model_h, model_w = last_frame_tensor.shape[-2:]

            scaled_boxes = ops.scale_boxes(
                (model_h, model_w), 
                preds[:, :4].clone(), 
                (orig_h, orig_w)
            )

            for i, box in enumerate(scaled_boxes):
                # Get the raw COCO class ID
                coco_cls = int(preds[i, 5])
                
                # === STEP 2: FILTER AND MAP ===
                if coco_cls in CLASS_MAPPING:
                    # Translate to DSEC ID
                    dsec_cls = CLASS_MAPPING[coco_cls]
                    
                    x1, y1, x2, y2 = box.tolist()
                    conf = float(preds[i, 4])

                    bbox = [
                        round(x1, 2), 
                        round(y1, 2), 
                        round(x2, 2), 
                        round(y2, 2)
                    ]
                    
                    boxes_to_save.append({
                        "class_id": dsec_cls, # Use the mapped ID
                        "conf": round(conf, 4),
                        "bbox": bbox
                    })
                    
        results_buffer[sequence_name].append({
            "frame_name": path_obj.name,
            "predictions": boxes_to_save
        })

    # 4. Save
    for seq_name, data in results_buffer.items():
        with open(output_dir / f"{seq_name}.json", 'w') as f:
            json.dump(data, f, indent=4)
            
    print(f"Vanilla inference finished. Results in {output_dir}")