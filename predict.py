import torch 

model = torch.hub.load("ultralytics/yolov5","custom","trained_model/best.pt")
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
img = "car_dataset/images/test/2015_02818.jpg"

# Inference 
results = model(img)

results.print()
results.show()


