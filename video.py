import cv2
import os

# Path to the folder containing images
image_folder = '/home/ashutosh/pulkit/yolo_temporal/runs/train/exp1/visualizations'  # <-- change this
output_video = '/home/ashutosh/pulkit/yolo_temporal/video/output.mp4'

# Get all image file names and sort them
images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
images.sort()  # ensure correct order

# Read the first image to get the frame size
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_frame.shape

# Define video writer (mp4, 30 fps)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

# Write each image as a frame
for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    frame = cv2.imread(img_path)
    if frame is not None:
        video.write(frame)

video.release()
cv2.destroyAllWindows()

print(f"✅ Video saved as {output_video}")
