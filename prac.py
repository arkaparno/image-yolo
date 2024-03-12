import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

import supervision as sv


class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names
    
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)
    

    def load_model(self):
       
        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8n model
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    

    # def plot_bboxes(self, results, frame):
        
    #     xyxys = []
    #     confidences = []
    #     class_ids = []
        
    #      # Extract detections for person class
    #     for result in results:
    #         boxes = result.boxes.cpu().numpy()
    #         class_id = boxes.cls[0]
    #         conf = boxes.conf[0]
    #         xyxy = boxes.xyxy[0]

    #         if class_id == 0.0:
          
    #           xyxys.append(result.boxes.xyxy.cpu().numpy())
    #           confidences.append(result.boxes.conf.cpu().numpy())
    #           class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
            
        
    #     # Setup detections for visualization
    #     detections = sv.Detections(
    #                 xyxy=results[0].boxes.xyxy.cpu().numpy(),
    #                 confidence=results[0].boxes.conf.cpu().numpy(),
    #                 class_id=results[0].boxes.cls.cpu().numpy().astype(int),
    #                 )
        
    
    #     # Format custom labels
    #     self.labels = [f"{self.CLASS_NAMES_DICT[detection[2]]} {detection[1]:0.2f}"
    #            for detection in detections]

        
    #     # Annotate and display frame
    #     frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        
    #     return frame
    # def plot_bboxes(self, results, frame):
    # # Check if results is not empty and has the expected structure
    #     if not results or not hasattr(results, 'xyxy'):
    #         print("No detection results with the expected format.")
    #         return frame

    #     # Assuming the correct results structure now, iterate through detections
    #     labels = []
    #     for i, det in enumerate(results):  # Assuming results is a list of detections
    #         if not det:
    #             continue  # Skip if det is empty or None
    #         for *xyxy, conf, cls_id in det:
    #             class_id = int(cls_id)  # Convert class ID to int
    #             confidence = float(conf)  # Ensure confidence is float
    #             label = f"{self.CLASS_NAMES_DICT[class_id]} {confidence:.2f}"
    #             labels.append(label)
    #         # Add your bounding box drawing logic here
    #         # For example, using cv2.rectangle to draw bounding boxes on `frame`
    #         # And cv2.putText to add labels

    #     return frame
    def plot_bboxes(self, results, frame):
        if not results:
            print("No detections.")
            return frame

        # Assuming results is a list of detections
        for detection in results:
            # Extract bounding box coordinates, confidence, and class ID
            x_min, y_min, x_max, y_max, confidence, class_id = list(map(int, detection[:4])) + list(detection[4:])
            label = f"{self.CLASS_NAMES_DICT[class_id]} {confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Put label on the frame
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame



    
    
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
        while True:
          
            start_time = time()
            
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
    
detector = ObjectDetection(capture_index=0)
detector()