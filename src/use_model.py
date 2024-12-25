from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
current_dir = os.getcwd()
def main():
    trained_yolov8_model = YOLO(f"{current_dir}\\models\\last.pt")
    results_yolov8 = trained_yolov8_model.val(data=f"{current_dir}\\Worksite-Safety-Monitoring-1\\data_for_run.yaml", imgsz=640)
    results_yolov8.results_dict
if __name__ == "__main__":
    main()


