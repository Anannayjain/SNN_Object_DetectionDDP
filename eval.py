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
import warnings
from utils import *
import gc
import yaml

# path = "/raid/ee-udayan/uganguly/opticalflow/data/DSEC/train_images/interlaken_00_c/images/left/rectified"
# path = "/raid/ee-udayan/uganguly/opticalflow/data/DSEC_DET/test/interlaken_00_a"
# base_path = "/raid/ee-udayan/uganguly/opticalflow/data/DSEC/train_images"
# Dataset = [x for i in os.listdir(path) ]

# Get all folder names in the base path
# folder_names = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# Generate the full paths and store them in Dataset
# Dataset = [os.path.join(base_path, folder, "images/left/rectified") for folder in folder_names]

# print(Dataset)  # Print to verify the output



def process_dataset(
    dataset,
    model,
    yolo_model,
    bb_dataset,
    method="",
    compute_stride=None,
    model_name="flownets",
    init_stride = 30,
    save_video=False
):
    """
    Processes each item in the dataset, performing object detection, bounding box tracking,
    and optional video annotation.
    """
    if compute_stride is None:
        def compute_stride(prev_iou, curr_iou, current_stride):
            return current_stride
            
    bb_compare = {}
    store_time = {}
    result_frames = []
    optical_flow = []
    optical_flow_frame = []
    model_name = model_name
    init_stride = init_stride
    stride_list = [init_stride]
    
    for video_no,image_path in tqdm(enumerate(dataset), desc="Processing dataset", unit="item", position=0):
        start_time = time.time()
        num_frames = 0
        bboxes_list = []
        stride = init_stride
        prev_idx = -stride
        curr_idx = 0
        prev_iou = 1
        curr_iou = 1
        retrieval_time = 0
        last_end_time = start_time
        yolo_count = 0
        flow_count = 0
        detections = []
        prev_image = 0
        flow_flops = 0
        tot_flow_flops = 0
        profiling_time = 0
        flag = 1
        track_bbox = []
        name = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        track_bbox = []
        # track_bbox = np.array([bb_dataset[name][0][0],])
        # print(track_bbox)
        # print(type(track_bbox))
        
        # Detect objects and track bounding boxes
        for idx, image in tqdm(enumerate(get_images_from_directory(image_path)), total=len(os.listdir(image_path)), desc="Processing frames", position=1):
            
            end_time = time.time()
            retrieval_time +=end_time-last_end_time
            if method == "entire_yolo":
                _, detections = detect_objects(image, model, model_name = yolo_model)
                track_bbox = detections
                yolo_count+=1
                # track_bbox, _ = select_bbox_with_max_iou(track_bbox, detections)
            
            elif method == "cropped_yolo":
                yolo_count+=1
                if len(track_bbox) == 0:
                    _, detections = detect_objects(image, model, model_name = yolo_model)
                    track_bbox = detections
                else:
                    _, detections = detect_objects_with_cropping(model, image, track_bbox, model_name = yolo_model)
                    track_bbox = detections
                # track_bbox, _ = select_bbox_with_max_iou(track_bbox, detections)
                
            elif method == "optical_flow":
                if(prev_idx + stride == curr_idx):
                    ## YOLO
                    yolo_count+=1
                    prev_idx = curr_idx
                    _, detections = detect_objects(image, model, model_name = yolo_model)
                    curr_iou = calc_iou(track_bbox, detections)
                    track_bbox = detections
                    # track_bbox, _ = select_bbox_with_max_iou(track_bbox[0], detections)
                    # track_bbox = np.array([track_bbox,])
                    stride = compute_stride(prev_iou, curr_iou, stride)
                    stride_list.append(stride)
                    prev_iou = curr_iou
                else:
                    ## Optical Flow
                    flow_count+=1
                    # vel = get_vel(idx)
                    vel = [[0,0]]
                    # print(vel)
                    # model_name=model_name, ckpt_path=ckpt_path, device=device, down_sample=down_sample
                    # optical_flow, flops, extra_time = get_flow_of_box(track_bbox, prev_image, image, model_name, down_sample = 1)
                    optical_flow, flops, extra_time = get_optical_flow(prev_image, image, model_name, down_sample = 1)
                    profiling_time += extra_time
                    tot_flow_flops += flops
                    # print(idx)
                    track_bbox = np.array(update_bounding_boxes(optical_flow, track_bbox, vel))
            # print(track_bbox)
            bboxes_list.append(track_bbox[:,:4])
            

            # Optionally annotate and store frames
            if save_video:
                frame = annotate_frame(image, track_bbox, color = (255, 0 , 0))
                frame = annotate_frame(frame, BB_Dataset[list(BB_Dataset.keys())[video_no]][idx] , color = (0, 0 , 255))    
                # print(frame)        
                # break   
                result_frames.append(frame)
                if len(optical_flow)>0:
                    optical_flow_frame.append(optical_flow)
        
            prev_image = image
            num_frames += 1
            curr_idx+=1
            last_end_time = time.time()

        end_time = time.time()
        # retrieval_time = num_frames * 8.25 / 1000
        yolo_time = (end_time - start_time) - retrieval_time - profiling_time
        fps_incl = num_frames / (end_time - start_time) if (end_time - start_time) > 0 else 0
        fps_excl = num_frames / yolo_time if yolo_time > 0 else 0
        
        yolo_flops = model.info()[3]
        print(yolo_flops*1e9)
        if(flow_count):
            print(tot_flow_flops/flow_count)
        # print(yolo_count)
        # print(flow_count)
        flops = (tot_flow_flops + yolo_count*yolo_flops*1e9) / (flow_count + yolo_count)

        
        parts = image_path.split(os.sep)
        sequence_name = parts[8]
        bb_compare[sequence_name] = bboxes_list
        # correct_bbox = np.array(bb_dataset[os.path.basename(os.path.dirname(os.path.dirname(image_path)))], dtype=float)
        # bb_compare = []
        correct_bbox = []
        # Store timing information
        store_time[os.path.basename(os.path.dirname(os.path.dirname(image_path)))] = {
            'num_frames': num_frames,
            'total_time': end_time - start_time,
            'retrieval_time': retrieval_time,
            'yolo_time': yolo_time,
            'fps_including_retrieval': fps_incl,
            'fps_excluding_retrieval': fps_excl,
            "flops": flops
        }

    return {
        'bb_compare': bb_compare,
        'store_time': store_time,
        'frames': result_frames,
        'stride': np.array(stride_list),
        'optical_frames' : optical_flow_frame
    }


# result[name] = process_dataset(
#     Dataset,
#     model,
#     yolo_model,
#     BB_Dataset,
#     method=method,
#     compute_stride=compute_stride,
#     # model_name="flownets",
#     model_name = "flownets",
#     # model_name = "no",
#     init_stride = 30,
#     save_video=True
# )
# print(result[name]["store_time"])
# print(np.mean(result[name]["stride"]))
# for i in BB_Dataset.keys():
#     print(i)
#     # iou, mAP, iou_list = get_eval_metric_dsec(i, result[name]['bb_compare'][i], BB_Dataset[i])
#     # print(i, iou, mAP, np.mean(iou_list))
#     iou, mAP, iou_list = get_eval_metric_dsec("anything", result[name]['bb_compare'][i], BB_Dataset[i])
#     # for map the order should be pred,true box
#     #ineed to change here
#     print(f"iou:{iou}, map:{mAP}, iou_list_mean: {np.mean(iou_list)}")
    
# result[name]['frames']



# save_video = False

# if(save_video):
#     # Define the output file paths
#     output_path_main = "main.mp4"
#     output_path_main_OF = "main_OF.mp4"

#     # Remove existing files if they exist
#     if os.path.exists(output_path_main):
#         os.remove(output_path_main)

#     if os.path.exists(output_path_main_OF):
#         os.remove(output_path_main_OF)

#     if save_video:
#         result[name]["frames"] = np.array(result[name]["frames"])
#         save_rgb_frames_to_video(result[name]["frames"], output_path_main, fps=30)

#         result[name]["optical_frames"] = np.array(result[name]["optical_frames"])
#         save_rgb_frames_to_video(result[name]["optical_frames"], output_path_main_OF, fps=30)

if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
base_path = config['dataset']['test']['path']
# Get all folder names in the base path
folder_names = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# Generate the full paths and store them in Dataset
Dataset = [os.path.join(base_path, folder, "images/left/rectified") for folder in folder_names]
print(Dataset)
# print("Number of Taken Videos: " ,len(Dataset), "\nTotal Images/Frames in Video: ", len(os.listdir(base_path)))
BB_Dataset = create_bb_dataset_dsec(Dataset)
## Config:
yolo_model = "v11"
# method = "entire_yolo"
method = "optical_flow"
compute_stride = None


if yolo_model == "v11":
    model = import_v11()
elif yolo_model == "v5":
    model = import_v5()

# Suppress all FutureWarnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)



error_x = []
error_y = []
result = {}

if compute_stride is None:
    name = method
else:
    name = method + "_dynamic"
