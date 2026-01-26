import numpy as np
import cv2
import sys
from typing import Optional
from sklearn.metrics import auc
import time
from tqdm.notebook import tqdm
import glob
import torch
import logging
from ultralytics import YOLO
import os
# import ptlflow
# from ptlflow.utils.io_adapter import IOAdapter
# from ptlflow.utils import flow_utils
from collections import Counter

_model_cache = {}


def create_bb_dataset_dsec(Dataset):
    """
    Creates a dictionary mapping frame indices (0 to 536) to lists of bounding boxes for the DSEC dataset.

    Args:
        bounding_boxes_list (list of lists): Each entry contains 
            [t, x, y, h, w, class_id, class_confidence, track_id].

    Returns:
        dict: A dictionary where keys are frame indices and values are lists of bounding boxes.
    """
    bb_dataset = {}
    
    timestamp_to_frame = {}  # Mapping timestamps to frame indices

    # Load the tracks.npy file
    ans = {}
    for i,path in enumerate(Dataset):
        
        # path = "/raid/ee-udayan/uganguly/opticalflow/data/DSEC/train_images/zurich_city_04_b/images/left/rectified"
        base_dir = os.path.dirname(os.path.dirname(path))
        timestamp_file = glob.glob(os.path.join(base_dir, "timestamps.txt"))

        parts = path.split(os.sep)
        # print(parts)
        sequence_name = parts[7]
        train_or_test = parts[6]

        tracks_path = os.path.join(r"/", os.path.join(*parts[:6]), train_or_test, sequence_name, "object_detections", "left", "tracks.npy")
        tracks = np.load(tracks_path, allow_pickle=True)
        
        # Initialize an empty list to store bounding boxes
        bounding_boxes_list = []
        
        # Iterate through each sequence in the dataset
        for sequence in tracks:
            # Each sequence is expected to have bounding boxes
            # Convert the bounding boxes to a list and append
           
            t,x,y,h,w,class_id,class_confidence,track_id = sequence
            bounding_boxes_list.append([t, x,y,h,w,class_id,class_confidence, track_id])

        
        total_frames = 0

        # Read the file and populate the dictionary
        with open(timestamp_file[0], "r") as file:
            for index, line in enumerate(file):
                timestamp = int(line.strip())  # Convert to integer
                timestamp_to_frame[timestamp] = index  # Store index as value
                total_frames+=1
                
        bboxes = [[] for _ in range(total_frames)]

        for box in bounding_boxes_list:
            frame_idx = timestamp_to_frame[box[0]]  # Get corresponding frame index
            # x1*2.13, y1*2.7, x2*2.13, y2*2.7
            bbox = [box[1]*2.13, box[2]*2.7, (box[3] + box[1])*2.13, (box[4] + box[2])*2.7, box[6], box[5], box[7]]  # Extract x1, y1, x1 + h, y1 + w, confidence, class_id, track_id
            bboxes[frame_idx].append(bbox)  # Append bbox to the corresponding frame

        bb_dataset[sequence_name] = bboxes
    return bb_dataset

def import_v5():    
    # Reduce log level for specific libraries
    logging.getLogger("torch").setLevel(logging.INFO)
    
    # Check and set up the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the YOLOv5 model
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True, _verbose=False)  # load silently)
        model.to(device).eval()  # Move model to the appropriate device and set to eval mode
        print("YOLOv5 model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
    
    # Set the NMS and inference parameters
    model.conf = 0.55  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = True  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    
    ## Can be set to cars and trucks for our use case
    model.classes = None  # Filter by class (e.g., [0, 15, 16] for COCO categories)

    model.max_det = 1000  # Maximum number of detections per image
    # model.amp = False  # Automatic Mixed Precision (AMP) inference
    return model

def import_v11():
    verbose = False
    # Reduce log level for specific libraries
    # logging.getLogger("torch").setLevel(logging.INFO)
    
    # Check and set up the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
        # Configure logging level
    log_level = logging.INFO if verbose else logging.ERROR
    logging.getLogger("torch").setLevel(log_level)
    logging.getLogger("yolov11").setLevel(log_level)  # Adjust for YOLO logs

    
    try:
        model = YOLO("yolo11m.pt")
        model.to(device).eval()  # Move model to the appropriate device and set to eval mode
        print("YOLOv11 model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
    
    # Set the NMS and inference parameters
    model.conf = 0.55  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = True  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    
    ## Can be set to cars and trucks for our use case
    model.classes = None  # Filter by class (e.g., [0, 15, 16] for COCO categories)
    
    model.max_det = 1000  # Maximum number of detections per image
    # model.amp = False  # Automatic Mixed Precision (AMP) inference
    return model


def read_image(image_path: str, convert_to_rgb: bool = True):
    """
    Read an image from a file path and optionally convert it to RGB format.

    Args:
        image_path (str): Path to the image file.
        convert_to_rgb (Optional[bool]): Whether to convert the image to RGB format. Default is True.

    Returns:
        image: The image in BGR or RGB format depending on the convert_to_rgb flag.

    Raises:
        FileNotFoundError: If the image cannot be read from the file path.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Error: Could not open image at {image_path}")
        raise FileNotFoundError(f"Error: Could not open image at {image_path}")
    if convert_to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_images_from_directory(directory_path):
    """
    Get images from the specified directory in sorted order.

    Args:
        directory_path (str): Path to the directory containing images.

    Yields:
        Processed image results one at a time.
    """
    image_files = sorted(
        [entry.path for entry in os.scandir(directory_path) if entry.is_file() and entry.name.endswith(".png")],
        key=lambda entry: int(os.path.splitext(os.path.basename(entry))[0])  # Sort by filename number
    )
    
    for file_path in image_files:
        yield read_image(file_path)


def detect_objects(frame, model, draw_bboxes=True, bbox_color=(255, 0, 0), bbox_thickness=2, model_name = "v11"):
    """
    Detect objects in an RGB or grayscale frame using the YOLOv5 model and optionally draw bounding boxes.

    Args:
        frame (np.ndarray): Input image frame (RGB or grayscale).
        model: Pre-trained YOLOv5 object detection model.
        draw_bboxes (bool): Whether to draw bounding boxes on the frame.
        bbox_color (tuple): Color of the bounding boxes (RGB format).
        bbox_thickness (int): Thickness of the bounding box lines.

    Returns:
        tuple: Processed frame with bounding boxes (if enabled) and detected bounding box data.

    Raises:
        ValueError: If the provided model is not a YOLOv5 model.
    """
    
# TODO: Ensure the model is a YOLOv5 model
    
    # if not hasattr(model, 'names') or not hasattr(model, 'yaml') or 'yolov5' not in str(model):
    #     raise ValueError("The provided model is not a YOLOv5 model. Please use the YOLOv5 model from 'ultralytics/yolov5'.")

    # Check input validity
    if frame.ndim == 2:  # Grayscale image
        img = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB for the model
    elif frame.ndim == 3 and frame.shape[2] == 3:  # RGB image
        img = frame
    else:
        raise ValueError("Invalid frame format: expected 2D grayscale or 3D RGB image.")

    # Run object detection model
    if model_name == "v5":
        results = model(img)
        bboxes = results.xyxy[0].cpu().numpy()  # Bounding box results
    else:
        results = model(img, verbose=False)
        bboxes = results[0].boxes.xyxy.cpu()  # Bounding box results
        cls_id = results[0].boxes.cls.cpu().unsqueeze(1)  # Add singleton dimension
        _conf = results[0].boxes.conf.cpu().unsqueeze(1)  # Add singleton dimension
        bboxes = torch.cat((bboxes, _conf, cls_id), dim=1).numpy()  # Concatenate and convert to numpy


    if draw_bboxes:
        # Prepare a copy of the original frame for visualization
        img_with_bboxes = frame.copy()
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, confidence, class_idx = bbox[:6]
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

            class_name = model.names[int(class_idx)]
            label = f"{class_name} {confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(img_with_bboxes, (x_min, y_min), (x_max, y_max), bbox_color, bbox_thickness)
            
            # Optional: Add label text
            cv2.putText(
                img_with_bboxes, 
                label, 
                (x_min, y_min - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                bbox_color, 
                1
            )
    else:
        img_with_bboxes = frame  # Return the original frame if no drawing is needed

    return img_with_bboxes, bboxes

def calc_iou(track_bbox, detections):
    """
    Computes the IoU between two lists of boxes.

    Args:
        boxes1: A list of bounding boxes, where each box is a tuple or list (x1, y1, x2, y2).
        boxes2: A list of bounding boxes, where each box is a tuple or list (x1, y1, x2, y2).

    Returns:
        A list of IoU values for all cross terms between the two lists of boxes.
    """
    if ((len(detections) == 0) or (len(track_bbox) == 0)):
        return 0
        
    top_n = min(len(track_bbox), len(detections))
    
    iou_matrix = compute_iou_list(np.array(detections)[:, :4], np.array(track_bbox)[:, :4])
    iou_list = iou_matrix.flatten()
    sorted_values = np.sort(iou_list)[::-1]
    
     # Take the top N values and compute average
    return float(np.mean(sorted_values[:top_n]))
    

def get_vel(idx):
    if idx == 0:
        return 0
    # Get bounding boxes for two consecutive frames
    boxes_t0 = BB_Dataset['interlaken-00-c'][idx-1]  # List of boxes at t0
    boxes_t1 = BB_Dataset['interlaken-00-c'][idx]  # List of boxes at t1
    
    # Convert to dictionary for quick lookup (track_id -> bbox)
    bboxes_t0 = {box[-1]: box for box in boxes_t0}  # {track_id: bbox}
    bboxes_t1 = {box[-1]: box for box in boxes_t1}
    
    # Compute velocities for each tracked object
    velocities = {}
    
    for track_id in bboxes_t0.keys() & bboxes_t1.keys():  # Only process matched track_ids
        box_t0 = bboxes_t0[track_id]
        box_t1 = bboxes_t1[track_id]
        
        # Compute center coordinates
        x0, y0 = (box_t0[0] + box_t0[2]) / 2, (box_t0[1] + box_t0[3]) / 2
        x1, y1 = (box_t1[0] + box_t1[2]) / 2, (box_t1[1] + box_t1[3]) / 2
    
        # Compute velocity components
        vx, vy = x1 - x0, y1 - y0
    
        velocities[track_id] = (vx, vy)
    return velocities


def get_optical_flow(prev_img, curr_img, model_name: str, down_sample=1, ckpt_path="things", device=None):
    """
    Compute optical flow between two images using either Farneback, Lucas-Kanade, or a pre-trained model from PTLFlow.

    Parameters:
    - prev_img (np.array): Previous image/frame (HWC).
    - curr_img (np.array): Current image/frame (HWC).
    - model_name (str): The name of the optical flow model to use.
    - down_sample (int): Factor by which to downscale the images to improve performance.
    - ckpt_path (str): Path to the model checkpoint.
    - device (torch.device): The device (CPU/GPU) to run the model on.

    Returns:
    - flow_rescaled (np.array): Optical flow visualization (HWC).
    """
    # Ensure device is set
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resize images once to improve performance
    
    h, w = prev_img.shape[:2]
    scaled_h, scaled_w = int(h // down_sample), int(w // down_sample)
    scaled_prev_img = cv2.resize(prev_img, (scaled_w, scaled_h))
    scaled_curr_img = cv2.resize(curr_img, (scaled_w, scaled_h))

    # Initialize flow array
    flow = np.zeros((scaled_h, scaled_w, 2), dtype=np.float32)

    # Choose optical flow method
    if model_name == "farneback":
        flow = compute_farneback_flow(scaled_prev_img, scaled_curr_img)
        extra_time = 0
        flops = 1e9
        
    elif model_name == "lucas_kanade":
        flow = compute_lucas_kanade_flow(scaled_prev_img, scaled_curr_img)
    elif model_name == "no":
        extra_time = 0
        flops = 0
        
        pass
    else:
        # Check if model is cached to avoid reloading
        if model_name not in _model_cache:
            model = ptlflow.get_model(model_name, ckpt_path=ckpt_path)
            model.to(device).eval()
            _model_cache[model_name] = model
        else:
            model = _model_cache[model_name]      

        # Compute optical flow using PTLFlow
        flow, flops, extra_time = compute_ptlflow(model, scaled_prev_img, scaled_curr_img, device)
        
    # Scale the flow back to the original size
    flow_rescaled = cv2.resize(
        flow * down_sample, 
        (w, h), 
        interpolation=cv2.INTER_NEAREST  # or cv2.INTER_AREA
    )
    
    return flow_rescaled, flops, extra_time


def compute_farneback_flow(prev_img, curr_img):
    """ Compute optical flow using the Farneback method. """
    prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
    curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_img_gray, curr_img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def compute_ptlflow(model, prev_img, curr_img, device):
    io_adapter = IOAdapter(model, prev_img.shape[:2])
    inputs = io_adapter.prepare_inputs([prev_img, curr_img])
    inputs['images'] = inputs['images'].to(device)

    start_time = time.time()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        with torch.profiler.record_function("model_inference"):
            model_start_time = time.time()
            predictions = model(inputs)
            model_end_time = time.time()
    key_averages = prof.key_averages()
    flops = 0
    for k in key_averages:
        flops += k.flops
    end_time = time.time()

    extra_time = (end_time - start_time) - (model_end_time - model_start_time)
        
    # Perform the inference and extract flow
    
    flow = predictions["flows"][0, 0]  # Remove batch and sequence dimensions
    flow = flow.permute(1, 2, 0).detach().cpu().numpy()  # Convert from CHW to HWC format and move to CPU
    
    return flow, flops, extra_time

def update_bounding_boxes(flow, bboxes, vel):
    """
    Update bounding boxes based on optical flow displacement.

    Args:
        flow (numpy.ndarray): Optical flow array of shape (H, W, 2), 
                              where the last dimension contains [flow_x, flow_y].
        bboxes (list of lists): List of bounding boxes, 
                                where each bbox is [x_min, y_min, x_max, y_max].
    
    Returns:
        list of lists: Updated bounding boxes after applying average flow displacement.
    """
    if len(bboxes) == 0:  # If no bounding boxes are detected, return empty list
        return []

    shifted_bboxes = []
    flow_height, flow_width, _ = flow.shape
    # from ptlflow.utils.flow_utils import flow_to_rgb
    # frame = flow_to_rgb(flow)
   
    # frame = annotate_frame(frame, bboxes)
    for bbox in bboxes:
        avg_flow = []
        x_min, y_min, x_max, y_max = map(int, bbox[:4])  # Ensure integer coordinates
        
        # Clip bounding box coordinates to avoid indexing out of bounds
        x_min = np.clip(x_min, 0, flow_width - 1)
        x_max = np.clip(x_max, 0, flow_width - 1)
        y_min = np.clip(y_min, 0, flow_height - 1)
        y_max = np.clip(y_max, 0, flow_height - 1)
        
        # Extract flow region and calculate mean flow
        flow_region = flow[y_min:y_max, x_min:x_max]

        ###############################33
        # print(vel[114])
        # print(type(vel[114]))
        # print("x")
        
        # flow_values = flow_region[..., 0].flatten()
        # counts, bin_edges = np.histogram(flow_values, bins=50)
        # max_idx = np.argmax(counts)  # Index of max frequency bin
        # highest_freq_x = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2  # Bin center
        # avg_flow.append(highest_freq_x)
        
        
        
        # Plot histogram
        # plt.figure(figsize=(8, 6))
        # plt.axvline(x=vel[114][0], color='red', linestyle='dashed', linewidth=2, label=f"x = {vel[114][0]:.2f}")
        # plt.hist(flow_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
        # plt.axvline(x=highest_freq_x, color='green', linestyle='dashed', linewidth=2, label=f"Mode: {highest_freq_x:.2f}")
        # plt.xlabel("Flow Values")
        # plt.ylabel("Frequency")
        # plt.title("Histogram of Flow Region")
        # plt.grid(True)
        # plt.show() 

        # print("y")

        # flow_values = flow_region[..., 1].flatten() 
        # counts, bin_edges = np.histogram(flow_values, bins=50)
        # max_idx = np.argmax(counts)  # Index of max frequency bin
        # highest_freq_x = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2  # Bin center
        # avg_flow.append(highest_freq_x)
        
        # Plot histogram
        # plt.figure(figsize=(8, 6))
        # plt.axvline(x=vel[114][1], color='red', linestyle='dashed', linewidth=2, label=f"x = {vel[114][1]:.2f}")
        # plt.hist(flow_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
        # plt.axvline(x=highest_freq_x, color='green', linestyle='dashed', linewidth=2, label=f"Mode: {highest_freq_x:.2f}")
        # plt.xlabel("Flow Values")
        # plt.ylabel("Frequency")
        # plt.title("Histogram of Flow Region")
        # plt.grid(True)
        # plt.show() 
        ############################333
        
        if(flow_region.size == 0):
            continue
            
        avg_flow = np.nan_to_num(np.mean(flow_region, axis=(0, 1), keepdims=False), nan=0).astype(int)
        # avg_flow = np.nan_to_num(np.nanmax(flow_region, axis=(0, 1), keepdims=False), nan=0).astype(int)
        # error_x.append(avg_flow[0]-vel[114][0])
        # error_y.append(avg_flow[1]-vel[114][1])

        

        # print(avg_flow)
        # print("avg:",avg_flow)
        # Shift bounding box by the average flow displacement
        shifted_bboxes.append([
            x_min + avg_flow[0], y_min + avg_flow[1],
            x_max + avg_flow[0], y_max + avg_flow[1]
        ])
    # plt.imshow(frame)  # No cmap for RGB images
    # plt.show()
    # print(bboxes)
    # print(shifted_bboxes)
    return shifted_bboxes

def annotate_frame(frame, bboxes, color = (0, 0, 255)):
    """
    Annotate the frame with bounding boxes.
    
    Parameters:
    - frame: The input frame to annotate.
    - bboxes: List of bounding boxes, each in the format [x_min, y_min, x_max, y_max].
    - color: The color of the bounding boxes in BGR format (default is blue).
    
    Returns:
    - Annotated frame.
    """
    
    annotated_frame = frame.copy()
    if len(bboxes)==0:
        return annotated_frame
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox[:4]
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 2)
    return annotated_frame

def compute_iou_list(bboxes1, bboxes2):
    """
    Computes the IoU between two sets of bounding boxes.
    boxes1: (N, 4) array, where each row is [x1, y1, x2, y2]
    boxes2: (M, 4) array, where each row is [x1, y1, x2, y2]
    Returns an (N, M) array of IoU values.
    """
    bboxes1 = bboxes1[:, None, :]  # shape (N, 1, 4)
    bboxes2 = bboxes2[None, :, :]  # shape (1, M, 4)

    # Extract coordinates: (N, M) after broadcasting
    x1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    y1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    x2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    y2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    # Compute intersection area
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Area of each bounding box (broadcast properly)
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    # Union area
    union_area = area1 + area2 - inter_area

    # IoU matrix (N, M)
    iou_matrix = inter_area / np.clip(union_area, a_min=1e-8, a_max=None)
    return iou_matrix

def calc_iou_eval(track_bbox, detections):
    """
    Computes the IoU between two lists of boxes.

    Args:
        boxes1: A list of bounding boxes, where each box is a tuple or list (x1, y1, x2, y2).
        boxes2: A list of bounding boxes, where each box is a tuple or list (x1, y1, x2, y2).

    Returns:
        A list of IoU values for all cross terms between the two lists of boxes.
    """
    if ((len(detections) == 0) or (len(track_bbox) == 0)):
        return 0
        
    top_n = len(detections)
    
    iou_matrix = compute_iou_list(np.array(detections)[:, :4], np.array(track_bbox)[:, :4])

    iou_list = []
    for i in range(iou_matrix.shape[0]): #iterate through each detection
        iou_list.append(np.max(iou_matrix[i, :])) #find the max iou for each detection

    return iou_list

def get_eval_metric_dsec(name, BB_Compare, BB_Dataset, iou_threshold=0.5):
    """
    Calculates the average IoU and mean average precision (mAP) for a set of bounding box comparisons.

    Args:
        name: A string identifier for the evaluation.
        BB_Compare: A list of lists of tracked bounding boxes.  Each inner list corresponds to a frame.
                     Each bounding box is a list or tuple (x1, y1, x2, y2, ...).
        BB_Dataset: A list of lists of detected bounding boxes. Each inner list corresponds to a frame.
                    Each bounding box is a list or tuple (x1, y1, x2, y2, confidence, ...).
        iou_threshold: The IoU threshold for considering a detection a true positive.  Defaults to 0.5.

    Returns:
        A tuple containing the average IoU and mAP.  Returns (0, 0) if there are no bounding boxes to compare.
    """
    if len(BB_Compare) != len(BB_Dataset):
        print("Error: BB_Compare and BB_Dataset have different lengths.")
        return "Error"

    all_iou_scores = []
    all_precisions = []
    avg_iou_list = []

    for i in range(len(BB_Compare)):
        
        iou_list = calc_iou_eval(BB_Compare[i], BB_Dataset[i])  # Compare track bboxes to detections
        all_iou_scores.extend(iou_list)
        avg_iou_list.append(np.mean(iou_list))
        # print(iou_list)
        # Calculate precision for this frame
        if iou_list:
            true_positives = np.sum(np.array(iou_list) >= iou_threshold)
            precision = true_positives / len(iou_list)
            all_precisions.append(precision)
        else:
            all_precisions.append(0)  # No detections in this frame

    if not all_iou_scores:
        return 0, 0  # Handle the case where there are no bounding boxes

    avg_iou = np.mean(all_iou_scores)
    mAP = np.mean(all_precisions) if all_precisions else 0  # Mean of the precisions

    return avg_iou, mAP, avg_iou_list

def save_rgb_frames_to_video(frames, output_path, fps=30):
    """
    Save a sequence of RGB frames to a video file.
    
    Parameters:
    - frames: numpy array of shape (num_frames, height, width, 3), where the last dimension is RGB.
    - output_path: path to the output video file.
    - fps: frames per second of the video.
    """
    
    
    
    num_frames, height, width, _ = frames.shape  # Unpack the shape of the frame array
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    
    # Iterate through each frame in the array
    for frame in frames:
        if _ == 2:
            from ptlflow.utils.flow_utils import flow_to_rgb
            frame = flow_to_rgb(frame)
        
        # Ensure the frame is in the correct format (uint8 type and values between 0-255)
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame
        video_writer.write(frame)
    
    # Release the video writer when done
    video_writer.release()
    
def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)