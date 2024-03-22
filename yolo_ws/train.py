from ultralytics import YOLO

# load  a detection model>
model = YOLO('yolov5s.pt')
model.train(data = 'data/train.yaml', epochs = 5, imgsz = 224, plots = True)