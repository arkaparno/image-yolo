import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Load the COCO class labels our YOLO model was trained on
labelsPath = "D:/Python_programs/YOLO/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# Load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
model = fasterrcnn_resnet50_fpn(pretrained=True).eval()  # Set the model to inference mode

# Initialize the video stream
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)

skip_frames = 5  # Process every 5th frame
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % skip_frames == 0:
        # Resize the frame to reduce processing time
        frame_small = cv2.resize(frame, (640, 480))

        # Convert the frame to a format suitable for PyTorch
        image = F.to_tensor(frame_small).unsqueeze(0)  # Add a batch dimension

        with torch.no_grad():
            predictions = model(image)

        # Draw the predictions on the frame
        for element in range(len(predictions[0]['boxes'])):
            boxes = predictions[0]['boxes'][element].cpu().numpy()
            score = np.round(predictions[0]['scores'][element].cpu().numpy(), decimals=4)
            if score > 0.8:  # Confidence threshold
                label = LABELS[predictions[0]['labels'][element] - 1]
                cv2.rectangle(frame_small, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (0, 255, 0), 2)
                cv2.putText(frame_small, label, (int(boxes[0]), int(boxes[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame_small)
    
    frame_count = frame_count + 1

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


