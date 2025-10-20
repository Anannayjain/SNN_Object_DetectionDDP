# import yaml
# import cv2
# import torch
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from pathlib import Path

# # Make sure your DSECDataset class is in a file named dataset.py
# # in the same directory as this script.
# from dataset import DSECDataset

# # Class names corresponding to the class_id in your dataset
# CLASS_NAMES = {
#     0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'bus',
#     4: 'truck', 5: 'bicycle', 6: 'motorcycle', 7: 'train'
# }
# # Colors for the bounding boxes for each class
# CLASS_COLORS = plt.cm.get_cmap('hsv', len(CLASS_NAMES))

# def visualize_and_save_sample(image_tensor, labels_tensor, sample_index, output_dir):
#     """
#     Visualizes the last frame of a sequence and saves the plot to a file.

#     Args:
#         image_tensor (torch.Tensor): The tensor containing the sequence of images.
#         labels_tensor (torch.Tensor): The tensor containing the labels for the last frame.
#         sample_index (int): The index of the sample for the plot title and filename.
#         output_dir (Path): The directory where the visualization will be saved.
#     """
#     # --- 1. Prepare the image ---
#     # Convert the PyTorch tensor (C, H, W) to a NumPy array
#     last_image_np = image_tensor[-1].numpy()
    
#     # --- FIX: Transpose the array to the Matplotlib format (H, W, C) ---
#     last_image_np = last_image_np.transpose((1, 2, 0))

#     # Denormalize if the image tensor was scaled to [0.0, 1.0]
#     if last_image_np.dtype != np.uint8:
#         last_image_np = (last_image_np * 255).clip(0, 255).astype(np.uint8)

#     # --- 2. Create the plot ---
#     fig, ax = plt.subplots(1, figsize=(12, 8))
#     ax.imshow(last_image_np)
#     ax.axis('off')

#     # --- 3. Draw bounding boxes and labels ---
#     if labels_tensor.numel() == 0:
#         ax.set_title(f"Sample {sample_index}: Last Frame (No Detections)")
#     else:
#         ax.set_title(f"Sample {sample_index}: Last Frame with Bounding Boxes")
#         for box in labels_tensor:
#             class_id, x_center, y_center, width, height = box.numpy()
#             class_id = int(class_id)
            
#             # Convert center coordinates to top-left corner
#             x1 = x_center - width / 2
#             y1 = y_center - height / 2
            
#             class_name = CLASS_NAMES.get(class_id, f'Unknown: {class_id}')
#             color = CLASS_COLORS(class_id / len(CLASS_NAMES))
            
#             # Create a Rectangle patch
#             rect = patches.Rectangle(
#                 (x1, y1), width, height,
#                 linewidth=2, edgecolor=color, facecolor='none'
#             )
#             ax.add_patch(rect)
            
#             # Add the class label text
#             ax.text(
#                 x1, y1 - 5, class_name,
#                 color='white', backgroundcolor=color,
#                 fontsize=9, bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=1)
#             )

#     # --- 4. Save the figure to a file instead of showing it ---
#     output_filename = output_dir / f"sample_{sample_index}.png"
#     plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
#     print(f"Saved visualization to: {output_filename}")
    
#     # Close the figure to free up memory
#     plt.close(fig)

# if __name__ == '__main__':
#     # Use a non-interactive backend for SSH
#     plt.switch_backend('Agg')

#     # --- Load Configuration ---
#     config_path = Path('config.yaml')
#     if not config_path.exists():
#         raise FileNotFoundError("Error: config.yaml not found. Please create it and update the dataset paths.")

#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)

#     # --- Create output directory ---
#     output_dir = Path("./visualizations")
#     output_dir.mkdir(exist_ok=True)
#     print(f"Visualizations will be saved in: {output_dir.resolve()}")

#     print("Loading dataset...")
#     try:
#         train_dataset = DSECDataset(config=config, mode='train')
#     except Exception as e:
#         print(f"\n--- Could not initialize dataset ---")
#         print(f"Error: {e}")
#         print("Please check that the 'path' in your config.yaml points to the correct directory.")
#         exit()

#     # --- Visualize a few random samples ---
#     num_samples_to_show = 3
#     if len(train_dataset) < num_samples_to_show:
#         print(f"Warning: Dataset has fewer than {num_samples_to_show} samples. Showing all {len(train_dataset)} samples.")
#         num_samples_to_show = len(train_dataset)

#     if num_samples_to_show > 0:
#         print(f"Generating visualizations for {num_samples_to_show} random samples...")
#         random_indices = random.sample(range(len(train_dataset)), num_samples_to_show)

#         for i in random_indices:
#             image_tensor, labels_tensor = train_dataset[i]
#             visualize_and_save_sample(image_tensor, labels_tensor, sample_index=i, output_dir=output_dir)
#     else:
#         print("No samples found in the dataset to visualize.")

