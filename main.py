import cv2
import numpy as np
import time
from logger import Logger

class YoloDetector:

    def __init__(self, weight_file, cfg_file):
        
        self.RECORD_COUNTER = self.get_record_counter('counter')
        self.NAMEFILE = "coco.names"
        self.OBJECT_LOG_NAME = 'bottle'
        self.LOGFILE = f"logs/records_{self.OBJECT_LOG_NAME}_{self.RECORD_COUNTER}.csv"


        # Set up network

        # self.net = cv2.dnn.readNet("nets/yolov4-leaky-416.weights", "nets/yolov4-leaky-416.cfg")
        self.net = cv2.dnn.readNet(weight_file, cfg_file)

        # Read class names from COCO dataset
        self.classes = []
        with open(self.NAMEFILE, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()

        # This line has either i[0] or i depending on the OpenCV version
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Logging utlity
        self.logger = Logger()

        # Colors for boxes in visualisation
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def get_record_counter(self, file):
        # Determine the record counter
        with open(file, 'r+', encoding='utf8') as f:
            lines = (line.strip() for line in f if line)
            x = [int(float(line.replace('\x00', ''))) for line in lines]
            ret = x[0]
            f.truncate(0)
            f.seek(0)
            f.write(str(ret + 1))

        return ret

    def detect_objects(self):

        # need CAP_V4L option here for Linux, potentially
        cap = cv2.VideoCapture(1)

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # print(frame_height, frame_width)

        output = cv2.VideoWriter(f'videos/output_{self.OBJECT_LOG_NAME}_{self.RECORD_COUNTER}.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
        font = cv2.FONT_HERSHEY_PLAIN
        starting_time = time.time()
        frame_id = 0


        # Video capture loop
        while (True):
            ret, frame = cap.read()
            frame_id += 1
            height, width, channels = frame.shape

            # Do magic ML stuff
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            # Set up detection logic
            class_ids = []
            confidences = []
            boxes = []
            records = np.empty((0,4))

            # Process detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.2:
                        # Object detected, find out the center
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                    
                        elapsed_time = time.time() - starting_time
                        confidences.append(float(confidence))
                        # Only collect logs for the object we are interested in
                        if str(self.classes[class_id]) == self.OBJECT_LOG_NAME:
                            record = np.array([center_x, center_y, 0, elapsed_time])
                            records = np.concatenate((records, [record,]))


                        # VISUALISATION
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        # Capture detection information
                        boxes.append([x, y, w, h])
                        class_ids.append(class_id)
                        # These frames per second are computed in a rolling fashion
                        fps = frame_id / elapsed_time

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

            # VISUALISATION
            # Draw boxes
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    confidence = confidences[i]
                    color = self.colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
                    cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, (255, 255, 255), 2)
                    print(label,confidence,x,y,w,h)
                    
                # Label boxes
                cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 2)
    
            # Log records and show frame
            self.logger.record_value(records)
            output.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Export logs, destroy window
        self.logger.export_to_csv(self.LOGFILE)
        cap.release()
        output.release()
        cv2.destroyAllWindows()


det = YoloDetector('nets/yolov4-leaky-416.weights', 'nets/yolov4-leaky-416.cfg')
det.detect_objects()