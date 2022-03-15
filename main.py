from turtle import position
import cv2
import numpy as np
import time
from logger import Logger


# Determine the record counter
text = open('counter.txt', 'r+')
RECORD_COUNTER = int(text.readline())
text.truncate(0)
text.write(str(RECORD_COUNTER + 1))

NAMEFILE = "coco.names"
OBJECT_LOG_NAME = 'bottle'
LOGFILE = f"logs/records_{OBJECT_LOG_NAME}_{RECORD_COUNTER}.csv"


net = cv2.dnn.readNet("nets/yolov4-leaky-416.weights", "nets/yolov4-leaky-416.cfg")

# Read class names from COCO dataset
classes = []
with open(NAMEFILE, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

# This line has either i[0] or i depending on the OpenCV version
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# need CAP_V4L option here for Linux, potentially
cap = cv2.VideoCapture(1)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# print(frame_height, frame_width)

output = cv2.VideoWriter(f'output_{OBJECT_LOG_NAME}_{RECORD_COUNTER}.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
logger = Logger()

# Video capture loop
while (True):
    ret, frame = cap.read()
    frame_id += 1
    height, width, channels = frame.shape

    # Do magic ML stuff
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Set up detection logic
    class_ids = []
    confidences = []
    boxes = []
    records = np.empty((0,4))

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected, find out the center
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                detected_position = (center_x, center_y, 0)

                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Capture detection information
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # These frames per second are computed in a rolling fashion
                elapsed_time = time.time() - starting_time
                fps = frame_id / elapsed_time

                # Only collect logs for the object we are interested in
                if str(classes[class_id]) == OBJECT_LOG_NAME:
                    record = np.array([center_x, center_y, 0, elapsed_time])
                    records = np.concatenate((records, [record,]))


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    
    # Draw boxes
    for i in range(len(boxes)):
      if i in indexes:
         x, y, w, h = boxes[i]
         label = str(classes[class_ids[i]])
         confidence = confidences[i]
         color = colors[class_ids[i]]
         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
         cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
         cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, (255, 255, 255), 2)
         print(label,confidence,x,y,w,h)
    
    # Label boxes
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 2)
    
    # Log records and show frame
    logger.record_value(records)
    output.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Export logs, destroy window
logger.export_to_csv(LOGFILE)
cap.release()
output.release()
cv2.destroyAllWindows()