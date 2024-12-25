import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ultralytics import YOLO


yolov8_model = YOLO("yolov8n.pt")
yolov8_results = yolov8_model.train(
    data="D:\CV2\Worksite-Safety-Monitoring-1\data.yaml",
    epochs = 2,
    batch = -1,
    optimizer = 'auto',
    device=0    
)