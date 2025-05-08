from ultralytics import YOLO

def load_student_model(weights_path):
    student = YOLO(weights_path)
    print(f"Student model loaded from {weights_path}")
    return student

if __name__ == "__main__":
    student = load_student_model("/Users/mahidharreddypatukuri/yolov8_image_detector/runs/detect/train/weights/best.pt")
