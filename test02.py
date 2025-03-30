from ultralytics import YOLO


# Run inference on an image
# yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt, yolo12n.pt
model = YOLO('yolo12n.pt')  # Load model

results = model('./videos/trafic02.avi', save=True)



