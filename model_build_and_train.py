from ultralytics import YOLO

# Load the pre-trained model
model = YOLO('yolov8s.pt')

# Train the model
model.train(
    data='/content/Violence_Detection_using_yoloV8/smart-schools-using-ai-and-iot3-1/data.yaml',
    epochs=12,
    imgsz=800,
    plots=True
)
