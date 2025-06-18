# YOLOv5 & YOLOv11 Object Detection Project

This project demonstrates the steps to train and predict using YOLOv5 and YOLOv11 on a custom dataset. The setup includes training a model, running inference on video files, and visualizing training logs with TensorBoard.

## Table of Contents

1. [Installation](#installation)
2. [Training the Model](#training-the-model)
3. [Running Inference on a Video](#running-inference-on-a-video)
4. [Viewing Training Logs with TensorBoard](#viewing-training-logs-with-tensorboard)

---

## Installation

1. **Clone the Repository**

   First, clone the YOLO repository for the model you wish to use:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   ```

2. **Set up a Python Environment**

   It’s recommended to use a virtual environment for this project.

   ```bash
   python -m venv yolovenv
   source yolovenv/bin/activate  # On Windows use yolovenv\Scriptsctivate
   ```

3. **Install Dependencies**

   Install the necessary dependencies for YOLO training and inference. This will install PyTorch, OpenCV, and other required packages:

   ```bash
   pip install -r requirements.txt
   ```

   If you need any additional libraries, you can also install them manually:
   ```bash
   pip install ultralytics opencv-python
   ```

---

## Training the Model

To train the YOLOv5 or YOLOv11 model, make sure your custom dataset is in the correct format (YOLO format), and the `dataset.yaml` file points to your training and validation data.

1. **YOLOv5 Training Command**

   Use the following command to train a YOLOv5 model. Adjust the `img`, `batch`, and `epochs` parameters as needed.

   ```bash
   python train.py --img 640 --batch 16 --epochs 100 --data dataset.yaml --weights yolov5s.pt --batch-size 4
   ```

   - `--img 640`: Image size (you can adjust this based on your hardware).
   - `--batch 16`: Batch size for training (you can tweak based on GPU memory).
   - `--epochs 100`: Number of training epochs.
   - `--data dataset.yaml`: Path to the dataset YAML file.
   - `--weights yolov5s.pt`: Pre-trained weights to start the training (optional).
   - `--batch-size 4`: Batch size for training.

2. **YOLOv11 Training Command**

   If you're using YOLOv11 for training, the following command can be used:

   ```bash
   yolo train model=yolo11n.pt data=dataset.yaml epochs=100 imgsz=640 batch=4
   ```

   - `model=yolo11n.pt`: The model you want to train.
   - `data=dataset.yaml`: Path to the dataset YAML file.
   - `epochs=100`: Number of epochs.
   - `imgsz=640`: Image size.
   - `batch=4`: Batch size.

After training, the best weights will be saved in `runs/train/exp/weights/best.pt` (or similar).

---

## Running Inference on a Video

Once your model is trained, you can use it to make predictions on a video.

1. **YOLOv5 Prediction Command**

   You can predict using the following command for YOLOv5:

   ```bash
   python detect.py --weights runs/train/exp2/weights/best.pt --source test.mp4 --img 640 --device 0
   ```

   - `--weights runs/train/exp2/weights/best.pt`: Path to the trained model weights.
   - `--source test.mp4`: Path to your video file.
   - `--img 640`: Image size.
   - `--device 0`: The device to run the inference on (0 for GPU, 'cpu' for CPU).

2. **YOLOv11 Prediction Command**

   Alternatively, for YOLOv11, use the following command to run predictions:

   ```bash
   yolo predict model=runs/detect/train3/weights/best.pt source=test.mp4 save=True device=0 imgsz=640
   ```

   - `model=runs/detect/train3/weights/best.pt`: Path to the trained model weights.
   - `source=test.mp4`: Path to your video file.
   - `save=True`: Save the output video with detections.
   - `device=0`: Use GPU device 0 (or `cpu` for CPU).
   - `imgsz=640`: Image size.

---

## Viewing Training Logs with TensorBoard

You can view the training logs to monitor your training progress with TensorBoard.

1. **Start TensorBoard**

   Use the following command to start TensorBoard:

   ```bash
   tensorboard --logdir runs/train/exp2
   ```

   - `--logdir runs/train/exp2`: Path to the training logs (this will vary based on your experiment).
   - After running the above command, you can navigate to `http://localhost:6006` in your browser to see the TensorBoard dashboard.

---

## Notes

- The paths used in the commands (e.g., `runs/train/exp2`) might vary depending on your specific setup and experiment number.
