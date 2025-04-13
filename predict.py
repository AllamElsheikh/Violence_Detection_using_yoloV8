from ultralytics import YOLO

# Define the external source (customize this path)
SOURCE = 'but the image path '  # <-- Replace this with your actual image/video/folder

# Load the trained YOLOv8 model
model = YOLO('/content/runs/detect/train/weights/best.pt')

# Run prediction
model.predict(
    source=SOURCE,
    conf=0.25,
    save=True,
    project='runs/detect',
    name='external_preds',
    exist_ok=True
)
