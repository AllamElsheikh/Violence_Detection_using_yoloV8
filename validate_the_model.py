from ultralytics import YOLO

# Dataset location class (same style)
class dataset:
    location = "/content/Violence_Detection_using_yoloV8/dataset"

# Load the trained model
model = YOLO('/content/runs/detect/train/weights/best.pt')

# Run validation
model.val(
    data=f'{dataset.location}/data.yaml'
)
