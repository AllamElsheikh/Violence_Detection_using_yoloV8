# 🚀 Violence Detection using YOLOv8

This project implements a YOLOv8-based object detection pipeline to detect violent actions in images and videos.  
It includes training, validation, and prediction scripts using the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) framework.

---

## 📁 Project Structure

Violence_Detection_using_yoloV8/ 
-  ├── laoding_data.py # Train the YOLOv8 model
-  ├── model_build_and_train.py # Validate the trained model
-  ├── predict.py # Predict on test set
-  ├── test_model.py # Predict on external data
-  ├── dataset/ # Contains images, labels, and data.yaml
-  └── validate_the_data.py/ # YOLOv8 training logs & predictions


## 🧠 Model

- **Base model**: `yolov8s.pt`
- **Custom dataset**: Prepared in YOLO format, configured in `data.yaml`
- **Training parameters**:
  - Epochs: 12
  - Image size: 800x800
  - Confidence threshold: 0.25

---

## 📦 Requirements

```bash
!pip install -r requirements.txt
```

## 🏋️‍♂️ loading the data
```bash
1python loading_data.py
```
## trainning 
``bash
!python model_build_and_train.py
```
## valid
```bash
!python validate_the_model.py
```
## predicting exteranl image / videos

```bash
!python predict.py
```



