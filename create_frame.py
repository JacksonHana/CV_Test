import os
import glob
import cv2
import shutil

output = "data/frames"
if os.path.exists(output):      # Check if the output directory exists
    shutil.rmtree(output)       # If it exists, delete the directory and its contents
os.makedirs(output)         # Create a new output directory

count = 0
for path in glob.iglob("*.mp4"):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        flag, frame = cap.read()    # Read a frame
        if not flag:                # If no frame is returned, the video has ended
            break
        cv2.imwrite("{}/{}.jpg".format(output, count), frame)  # Save the frame as a JPEG file
        count += 1                  # Increment the frame count 
    