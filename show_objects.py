import sys
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")
class_names = model.names  # Get the class name mapping
print(class_names)
sys.exit()

