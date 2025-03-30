import sys
import logging
import cv2
import platform
from ultralytics import YOLO


# Function to beep
def beep():
    if platform.system() == "Windows":
        # Beep at 1000 Hz for 200 ms
        winsound.Beep(1000, 200)
    else:
        # ASCII bell, may or may not work depending on terminal settings
        sys.stdout.write('\a')
        sys.stdout.flush()

# Function to process one frame and model results detected on the frame
def process_frame(frame, res, detect_y):
    # Process each detection result
    possible_colision = False
    for box, cls_id, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
        if int(cls_id) in [0, 60, 69]:
            continue

        x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
        class_name = class_names[int(cls_id)]
        confidence = float(conf)

        # Calculate center coordinates
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Optionally, add drawings to frame (bounding boxes and text for all detections)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)  # Blue dot
        label = f"{int(cls_id)} - {class_name}  ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)        

        # Check if the bottom of the detection crosses below the blue line
        if y2 > detect_y:
            possible_colision = True

    return frame, possible_colision



# On Windows, use winsound for beeps
if platform.system() == "Windows":
    import winsound

# Suppress Ultralytics warnings
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load the YOLO model
model = YOLO("yolo11n.pt")
class_names = model.names  # Get class name mapping
print(class_names)

# Process every nth frame for performance (adjust as needed)
frame_interval = 5

# Run YOLO on the webcam (source=0 for default camera; change if needed)
results = model.predict(source=0, stream=True, conf=0.6, vid_stride=frame_interval, verbose=False)

# loop over GENERATOR (infinite array of frames) provided by stream=True
for res in results:
    frame = res.orig_img  # Get original frame
    height, width, _ = frame.shape

    # Determine the y coordinate for the horizontal blue line (5% from the bottom)
    line_y = int(height * 0.95)

    # Draw the horizontal blue line (blue color in BGR)
    cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)

    # Flag to indicate if any detection is below the blue line

    frame, blockage = process_frame(frame, res, line_y)

    # If any detection is below the line, draw a filled red square (20x20 pixels)
    if blockage:
        cv2.rectangle(frame, (0, 0), (20, 20), (0, 0, 255), -1)
        beep()  # Emit a beep sound

    # Display the modified frame
    cv2.imshow("YOLO Detection with Alert", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
