from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train4/weights/best.pt")  # Adjust path to best.pt

# Perform prediction on a local image
results = model.predict(source=r"D:\c.jpg", save=True, imgsz=640)

# Perform prediction on a video
# results = model.predict(source="path/to/your/video.mp4", save=True)

# Perform prediction on a folder of images
# results = model.predict(source="path/to/your/folder", save=True)
