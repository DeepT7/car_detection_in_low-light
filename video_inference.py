import cv2 
import torch 

model = torch.hub.load('ultralytics/yolov5', 'custom', 'trained_model/best.pt')

color_dict = {
    1: (200, 0, 0),    # Class 1: Red
    2: (100, 100, 200),  # Class 2: Light Blue
    3: (0, 200, 100),   # Class 3: Green
    4: (120, 200, 0),   # Class 4: Lime Green
    5: (200, 100, 120),  # Class 5: Pink
    6: (120, 0, 120),   # Class 6: Purple
    7: (0, 0, 255),    # Class 7: Blue
    8: (0, 255, 0),    # Class 8: Green
    9: (100, 255, 0),   # Class 9: Lime
    10: (100, 10, 100),  # Class 10: Custom Color
    11: (0, 255, 210)   # Class 11: Custom Color
}
cap = cv2.VideoCapture('images/night_street.mp4')

frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter('output/output_video.avi', cv2.VideoWriter_fourcc(*'MPEG'),
                      30, (frame_width, frame_height))

while cap.isOpened():
    rec, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference 
    results = model(image)

    # Get bounding box 
    boxes, confidences, class_ids = results.pred[0][:, :4], results.pred[0][:, 4], results.pred[0][:, 5]

    # Drawing bounding boxes 
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = color_dict.get(int(class_id), (0, 0, 255)) # Default to read if class_id not found
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # cv2.putText(frame, f'Class {int(class_id)}', (x1, y1-10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    out.write(frame)
    cv2.imshow('video', frame)
    q = cv2.waitKey(90)
    if q == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()
out.release()