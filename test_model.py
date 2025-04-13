from ultralytics import YOLO

# Define dataset location
class dataset:
    location = "/content/Violence_Detection_using_yoloV8/dataset"

# Load the trained model
model = YOLO('/content/runs/detect/train/weights/best.pt')

# Run prediction
model.predict(
    source=f'{dataset.location}/test/images',
    conf=0.25,
    save=True
)
