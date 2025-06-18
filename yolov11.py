from ultralytics import YOLO

model =  YOLO("yolov11n.pt")  # Load a pre-trained YOLOv8 model

results = model.train(
    data="data.yaml",  # Path to the dataset configuration file
    epochs=100,        # Number of training epochs
    imgsz=640,         # Input image size
    batch=4 ,          # Batch size
    device=0,         # Device to train on (0 for GPU, 'cpu' for CPU)
    project="runs/train",  # Directory to save training results
    name="yolov11n_tray_dish"  # Name of the training run
)