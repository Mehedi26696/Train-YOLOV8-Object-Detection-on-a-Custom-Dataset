from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=50)  # train the model


# We can use this instead of this code in terminal: yolo task=detect mode=train model=yolov8n.yaml data=config.yaml epochs=1