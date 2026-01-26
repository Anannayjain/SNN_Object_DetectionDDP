import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import os  

def dsec_to_json(tracks_path, timestamps_path, output_path):
    """
    Converts DSEC .npy tracks to a JSON format
    """
    tracks = np.load(tracks_path)
    frame_timestamps = np.loadtxt(timestamps_path, usecols=0, dtype=np.int64)
    
    detection_ts = tracks['t']
    indices = np.searchsorted(frame_timestamps, detection_ts, side='left')
    indices = np.clip(indices, 0, len(frame_timestamps) - 1)
    
    ts_before = frame_timestamps[np.maximum(0, indices - 1)]
    ts_after = frame_timestamps[indices]
    
    final_indices = indices - (detection_ts - ts_before < ts_after - detection_ts)

    gt_list = []
    unique_frames = np.unique(final_indices)
    
    print(f"Processing {len(unique_frames)} frames for {output_path.name}...")
    
    for frame_idx in unique_frames:
        frame_name = f"{int(frame_idx):06d}.png"
        
        mask = (final_indices == frame_idx)
        frame_tracks = tracks[mask]
        
        if len(frame_tracks) == 0:
            continue

        frame_annotations = []
        
        for row in frame_tracks:
            x, y, w, h = row['x'], row['y'], row['w'], row['h']
            
            if w <= 0 or h <= 0: continue
            
            x1, y1 = float(x), float(y)
            x2, y2 = float(x + w), float(y + h)
            
            obj = {
                "class_id": int(row['class_id']),
                "conf" : 1,
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            }
            frame_annotations.append(obj)
            
        if frame_annotations:
            gt_list.append({
                "frame_name": frame_name,        
                "annotations": frame_annotations 
            })

    with open(output_path, 'w') as f:
        json.dump(gt_list, f, indent=4)

if __name__ == "__main__":
    BASE_DATASET_DIR = Path("/home/ashutosh/pulkit/SNN_scratch_AJ/dsec_dataset/test") 
    OUTPUT_DIR = Path("ground_truth")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sequence_dirs = [d for d in BASE_DATASET_DIR.iterdir() if d.is_dir()]
    sequence_dirs.sort() # Sort to keep processing order consistent

    print(f"Found {len(sequence_dirs)} sequences in {BASE_DATASET_DIR}")

    for seq_path in sequence_dirs:
        seq_name = seq_path.name

        tracks_path = seq_path / 'object_detections/left/tracks.npy'
        timestamps_path = seq_path / 'images/timestamps.txt'
        
        output_json_path = OUTPUT_DIR / f"{seq_name}.json"

        if not tracks_path.exists():
            print(f"Skipping {seq_name}: 'tracks.npy' not found.")
            continue
            
        if not timestamps_path.exists():
            print(f"Skipping {seq_name}: 'timestamps.txt' not found.")
            continue

        print(f"\nProcessing sequence: {seq_name}")
        try:
            dsec_to_json(tracks_path, timestamps_path, output_json_path)
        except Exception as e:
            print(f"Error processing {seq_name}: {e}")

    print(f"\nAll Done! Ground truth JSONs saved to: {OUTPUT_DIR.absolute()}")