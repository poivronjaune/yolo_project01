import logging
import cv2
from ultralytics import YOLO

# Suppress Ultralytics warnings
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load the YOLO model
model = YOLO("yolo11n.pt")
class_names = model.names  # Get class name mapping
print(class_names)

frame_interval = 5  # Process every nth frame

# Run YOLO on webcam (source=0 for default camera)
results = model.predict(source=0, stream=True, conf=0.6, vid_stride=frame_interval, verbose=False)

# Process frames
for res in results:
    frame = res.orig_img  # Get original frame

    # Draw a red diagonal line from top-right to bottom-left
    height, width, _ = frame.shape
    start_point = (width - 1, 0)  # Top-right corner
    end_point = (0, height - 1)   # Bottom-left corner
    color = (0, 0, 255)  # Red (BGR format)
    thickness = 2
    cv2.line(frame, start_point, end_point, color, thickness)

    # Draw bounding boxes around detected objects
    for box, cls_id, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        class_name = class_names[int(cls_id)]  # Get class name
        confidence = float(conf)

        if int(cls_id) in [0, 60]:
            continue

        # Draw the rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

        # Put the label text
        label = f"{int(cls_id)} - {class_name}  ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the modified frame
    cv2.imshow("YOLO Detection", frame)

    # Check for user quitting (press 'q' to exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()  # Close the display window
