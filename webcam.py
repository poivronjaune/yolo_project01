import sys
import logging
from ultralytics import YOLO

# Suppress Ultralytics warnings by setting logging level
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load the YOLO model
model = YOLO("yolo11n.pt")
class_names = model.names  # Get the class name mapping
#print(class_names)
#sys.exit()

# ====================== START DETECTING OBJECTS ================

id_to_detect = 65 # Remote (Print the class_names to change the id to detecte)

# Set the frame interval (process every nth frame) for performance improvement
frame_interval = 10  

# Run YOLO on the webcam (0 = default camera) and display annotations
results = model.predict(source=0, stream=True, show=True, vid_stride=frame_interval, verbose=False)

for res in results:
    detected_classes = res.boxes.cls.tolist()  # List of detected class IDs

    if id_to_detect in detected_classes:
        print(f"{class_names[id_to_detect]} detected!")
        break
