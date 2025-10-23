from pathlib import Path
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class DSECDataset(Dataset):
    def __init__(self, config, mode="train", transform=None):
        """
        Initializes the dataset.
        Args:
            config (dict): Configuration dictionary loaded from a YAML file.
            mode (str): If "train", loads the training dataset; if "test", loads the test dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid mode '{mode}'. Choose from 'train', 'val', or 'test'.")
        
        self.mode = mode
        self.transform = transform
        self.samples = []
        self.all_labels = {}

        data_config = config["dataset"][self.mode]
        path = Path(data_config["path"])
        self.sequence_length = data_config["seq_len"]

        sequence_dirs = [d for d in path.iterdir() if d.is_dir()]

        for seq_path in sequence_dirs:
            image_dir = seq_path / 'images/left/distorted'
            image_files = sorted(image_dir.glob('*.png'))
            num_images = len(image_files)

            timestamps_file = seq_path / 'images/timestamps.txt'
            frame_timestamps = np.loadtxt(timestamps_file, usecols=0, dtype=np.int64)
            
            # --- Labels are only needed for 'train' and 'val' modes ---
            if self.mode in ['train', 'val']:
                track_file = seq_path / 'object_detections/left/tracks.npy'
                tracks = np.load(track_file)
                # print(tracks[0])
                self.all_labels[str(image_dir)] = self._process_tracks(tracks, frame_timestamps)
            
            if num_images >= self.sequence_length:
                str_image_files = [f.name for f in image_files]
                for i in range(num_images - self.sequence_length + 1):
                    self.samples.append((image_dir, str_image_files, i))

        print(f"Dataset initialized with {len(self.samples)} total sequences.")


    def _process_tracks(self, tracks, frame_timestamps):
        """
        Processes the raw tracks numpy array to a dictionary mapping frame indices to bounding boxes.
        Handles the structured array format from Prophesee.

        Args:
            tracks (np.ndarray): The structured numpy array loaded from tracks.npy.
            frame_timestamps (np.ndarray): A sorted numpy array of image timestamps in microseconds.

        Returns:
            dict: A dictionary where keys are frame indices and values are lists of bounding boxes.
        """
        labels = {}
        detection_ts = tracks['t']
        # print(detection_ts)
        indices = np.searchsorted(frame_timestamps, detection_ts, side='left')
        # print(indices)

        indices = np.clip(indices, 0, len(frame_timestamps) - 1)
        ts_before = frame_timestamps[np.maximum(0, indices - 1)]
        ts_after = frame_timestamps[indices]
        final_indices = indices - (detection_ts - ts_before < ts_after - detection_ts)

        all_boxes = np.stack([
            tracks['class_id'].astype(np.float32),
            tracks['x'] + tracks['w'] / 2.0,
            tracks['y'] + tracks['h'] / 2.0,
            tracks['w'],
            tracks['h']
        ], axis=1)

        for i, frame_idx in enumerate(final_indices):
            if frame_idx not in labels:
                labels[frame_idx] = []
            labels[frame_idx].append(all_boxes[i])

        return labels

    def __len__(self):
        return len(self.samples)

    # def __getitem__(self, idx):        
    #     image_dir, image_files, start_idx = self.samples[idx]
        
    #     img_h, img_w = None, None
    #     sequence_images = []

    #     for i in range(self.sequence_length):
    #         img_path = str(image_dir / image_files[start_idx + i])
    #         image = cv2.imread(img_path)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
    #         if i == self.sequence_length - 1:
    #             img_h, img_w, _ = image.shape

    #         if self.transform:
    #             image = self.transform(image)
    #         else:
    #             image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    #         sequence_images.append(image)
        
    #     image_tensor = torch.stack(sequence_images)
        
    #     if self.mode in ['train', 'val']:
    #         target_frame_idx = start_idx + self.sequence_length - 1
    #         sequence_labels = self.all_labels.get(str(image_dir), {})
    #         target_labels = sequence_labels.get(target_frame_idx, [])
    #         if target_labels:                
    #             labels_np = np.array(target_labels, dtype=np.float32)
    #             labels_np[:, 1:] /= [img_w, img_h, img_w, img_h]
    #             labels_tensor = torch.tensor(labels_np)
    #         else:
    #             labels_tensor = torch.empty((0, 5), dtype=torch.float32)
    #         return image_tensor, labels_tensor
    #     else: # self.mode == 'test'
    #         last_frame_path = str(Path(image_dir) / image_files[start_idx + self.sequence_length - 1])
    #         return image_tensor, last_frame_path
    # In DSECDataset class in dataset.py

    def __getitem__(self, idx):      
        image_dir, image_files, start_idx = self.samples[idx]
        
        sequence_images = []
        img_h, img_w = None, None 
        
        for i in range(self.sequence_length):
            img_path = str(image_dir / image_files[start_idx + i])
            image = cv2.imread(img_path)
            
            if i == self.sequence_length - 1:
                # Store original dimensions for normalization
                img_h, img_w, _ = image.shape
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                image = self.transform(image)
            else:
                # Note: No resize here. Assumes all images are same (H, W)
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            sequence_images.append(image)
        
        image_tensor = torch.stack(sequence_images)
        
        if self.mode in ['train', 'val']:
            target_frame_idx = start_idx + self.sequence_length - 1
            sequence_labels = self.all_labels.get(str(image_dir), {})
            target_labels = sequence_labels.get(target_frame_idx, [])
            
            if target_labels:
                labels_array = np.array(target_labels, dtype=np.float32) # Shape (N, 5)
                # --- ADD A PRINT HERE ---
                # print(f"DEBUG: Frame {target_frame_idx} started with {labels_array.shape[0]} labels.")

                # 1. First-pass filter (from previous fix)
                mask = (labels_array[:, 3] > 0) & (labels_array[:, 4] > 0)
                labels_array = labels_array[mask]
                
                # Check if any valid labels remain
                if labels_array.shape[0] > 0 and img_h is not None and img_w is not None:
                    
                    # 2. Normalize
                    labels_array[:, 1] /= img_w  # cx
                    labels_array[:, 2] /= img_h  # cy
                    labels_array[:, 3] /= img_w  # w
                    labels_array[:, 4] /= img_h  # h

                    # --- START OF NEW FIX ---
                    
                    # 3. Clip out-of-bounds boxes
                    # Convert to xyxy for easier clipping
                    cx = labels_array[:, 1]
                    cy = labels_array[:, 2]
                    w = labels_array[:, 3]
                    h = labels_array[:, 4]
                    
                    x1 = np.clip(cx - w / 2, 0, 1)
                    y1 = np.clip(cy - h / 2, 0, 1)
                    x2 = np.clip(cx + w / 2, 0, 1)
                    y2 = np.clip(cy + h / 2, 0, 1)
                    
                    # Convert back to cxcywh
                    labels_array[:, 1] = (x1 + x2) / 2
                    labels_array[:, 2] = (y1 + y2) / 2
                    labels_array[:, 3] = (x2 - x1)
                    labels_array[:, 4] = (y2 - y1)
                    
                    # 4. Re-filter: Clipping might have created new zero-area boxes
                    mask = (labels_array[:, 3] > 0) & (labels_array[:, 4] > 0)
                    labels_array = labels_array[mask]
                    
                    # --- END OF NEW FIX ---

                    # --- ADDED THE SECOND PRINT STATEMENT HERE ---
                    # print(f"DEBUG: Frame {target_frame_idx} finished with {labels_array.shape[0]} labels.\n")
                    # ---------------------------------------------

                    # Check again if any valid labels remain after clipping
                    if labels_array.shape[0] > 0:
                        labels_tensor = torch.tensor(labels_array, dtype=torch.float32)
                    else:
                        labels_tensor = torch.empty((0, 5), dtype=torch.float32)
                
                else:
                    # No labels remained after first-pass filter
                    # print(f"DEBUG: Frame {target_frame_idx} finished with 0 labels (failed first filter).\n") # Added print here too
                    labels_tensor = torch.empty((0, 5), dtype=torch.float32)
            else:
                # No labels for this frame
                # print(f"DEBUG: Frame {target_frame_idx} finished with 0 labels (no target_labels).\n") # Added print here too
                labels_tensor = torch.empty((0, 5), dtype=torch.float32)
                
            return image_tensor, labels_tensor
        else: # self.mode == 'test'
            last_frame_path = str(Path(image_dir) / image_files[start_idx + self.sequence_length - 1])
            return image_tensor, last_frame_path

if __name__ == "__main__":
    import yaml
    import os

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    track_file = os.path.join('/home/ashutosh/pulkit/SNN_scratch_AJ/dsec_dataset/train/interlaken_00_c', 'object_detections/left/tracks.npy')            
    tracks = np.load(track_file)     

    print(tracks[0]['t'])