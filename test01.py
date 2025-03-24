from ultralytics import YOLO


# Run inference on an image
# yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt, yolo12n.pt
model = YOLO('yolo12n.pt')  # Load model
results = model('.\images\image02.jpg') 

# Process results list
for result in results:
    boxes = result.boxes                    # Boxes object for bounding box outputs
    masks = result.masks                    # Masks object for segmentation masks outputs
    keypoints = result.keypoints            # Keypoints object for pose outputs
    probs = result.probs                    # Probs object for classification probabilities outputs
    obb = result.obb                        # Oriented boxes object for oriented bounding boxes (OBB) outputs
    # result.show()                         # Show annotated image
    result.save(filename='result01.jpg')    # Save image with bounding boxes
