import os
import cv2
import glob
import shutil


image_folder = "Dataset/Detection/train/images"
label_folder = "Dataset/Detection/train/labels"

output_path = "data/frames"
if os.path.exists(output_path):      # Check if the output directory exists
    shutil.rmtree(output_path)       # If it exists, delete the directory and its contents
os.makedirs(output_path)         # Create a new output directory

def draw_bounding_boxes(image_path, label_path):
    img = cv2.imread(image_path)    # Read the image
    height, width, _ = img.shape  # Get the dimensions of the image

    # Read the label file
    with open(label_path, 'r') as file:
        for line in file:
            label, x_center, y_center, w, h = map(float, line.split())

            # Conver the normalized coordinates to pixel coordinates
            x_center = int(x_center * width)
            y_center = int(y_center * height)
            w = int(w * width)
            h = int(h * height)

            # Convert YOLO format to pixel coordinates
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)
            
            # Draw the bounding box on the image
            color = (0, 255, 0)  if label == 1 else (0, 0, 255) # Green for label 1 - tray, Red for label 0 - dish
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img

counter = 0
for img in os.listdir(image_folder):
    if img.endswith(".jpg") or img.endswith(".png"):
        label_name = os.path.splitext(img)[0] + ".txt"  # Assuming labels are in the same directory with .txt extension
        image_path = os.path.join(image_folder, img)
        label_path = os.path.join(label_folder, label_name)

        # Draw bounding boxes on the image
        output_img = draw_bounding_boxes(image_path, label_path)

        # Save the output image with bounding boxes
        # output_path = os.path.join(output_path, img)

        cv2.imwrite("{}/{}.jpg".format(output_path, counter), output_img)
        counter +=1

        print(f"Processed {img} and saved to {output_path}\n")
