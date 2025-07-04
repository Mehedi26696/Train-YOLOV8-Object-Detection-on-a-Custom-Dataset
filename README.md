# Alpaca Prediction Project

A deep learning project for detecting and predicting alpacas in images using YOLOv8 object detection model.

## Project Overview

This project implements an alpaca detection system using the YOLO (You Only Look Once) v8 model from Ultralytics. The model is trained to identify alpacas in images and can perform real-time detection on images, videos, or webcam feeds.

## Project Structure

```
Alpaca Prediction/
├── data/
│   ├── images/
│   │   └── train/          # Training images
│   └── labels/
│       ├── train.cache     # Training cache file
│       └── train/          # Training annotation files
├── prediction/
│   ├── config.yaml         # Dataset configuration
│   ├── train.py           # Training script
│   ├── predict.py         # Prediction script
│   └── runs/              # Training and prediction results
│       └── detect/
│           ├── train4/    # Latest training run
│           └── predict*/  # Prediction results
├── Process.txt            # Project workflow notes
└── README.md             # This file
```

## Features

- **Object Detection**: Detect alpacas in images using YOLOv8
- **Training Pipeline**: Custom training on alpaca dataset
- **Prediction Support**: Supports images, videos, and batch processing
- **Real-time Detection**: Fast inference for real-time applications

## Requirements

Before running this project, ensure you have the following installed:

```bash
pip install ultralytics
pip install torch torchvision
pip install opencv-python
pip install pillow
```

Or install using requirements.txt:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Step 1: Collect Dataset
- Download alpaca images from Google Images or other sources
- Ensure diverse poses, lighting conditions, and backgrounds

### Step 2: Annotation
- Use online annotation tool: [cvat.ai](https://cvat.ai)
- Create bounding box annotations around alpacas
- Export annotations in YOLO format

### Dataset Structure
The dataset should follow this structure:
```
data/
├── images/
│   └── train/          # All training images (.jpg format)
└── labels/
    └── train/          # Corresponding annotation files (.txt format)
```

## Configuration

The `config.yaml` file contains dataset configuration:

```yaml
path: D:\Depression\Deep Learning Projects\Aplaca Prediction\data
train: images/train
val: images/train

# Classes
names:
  0: alpaca
```

## Training

### Method 1: Using Python Script

Run the training script:

```bash
cd prediction
python train.py
```

### Method 2: Using Terminal Command

```bash
yolo task=detect mode=train model=yolov8n.yaml data=config.yaml epochs=50
```

### Training Parameters
- **Model**: YOLOv8 Nano (yolov8n.yaml)
- **Epochs**: 50 (adjustable for better accuracy)
- **Task**: Object Detection
- **Data**: Custom alpaca dataset

## Prediction

### Using Python Script

```bash
cd prediction
python predict.py
```

The prediction script supports:
- **Single Image**: Predict on a specific image
- **Video**: Process video files
- **Batch Processing**: Process entire folders

### Prediction Examples

```python
# Single image prediction
results = model.predict(source="path/to/image.jpg", save=True, imgsz=640)

# Video prediction
results = model.predict(source="path/to/video.mp4", save=True)

# Folder prediction
results = model.predict(source="path/to/folder/", save=True)
```

## Results

- Training results are saved in `prediction/runs/detect/train*/`
- Prediction results are saved in `prediction/runs/detect/predict*/`
- Best model weights: `prediction/runs/detect/train4/weights/best.pt`

## Model Performance

The model achieves good performance on alpaca detection tasks. For optimal results:
- Use more training epochs (50+ recommended)
- Ensure diverse training data
- Consider data augmentation techniques

## Usage Tips

1. **Improve Accuracy**: Increase epochs for better model performance
2. **Dataset Quality**: Use high-quality, diverse images for training
3. **Annotation Precision**: Ensure accurate bounding box annotations
4. **Model Selection**: Consider larger YOLO models (yolov8s, yolov8m) for better accuracy

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Ensure proper PyTorch GPU installation
2. **Path Errors**: Update file paths in config.yaml and scripts
3. **Memory Issues**: Reduce batch size or use smaller model variants

### Performance Optimization

- Use GPU acceleration for faster training/inference
- Optimize image input size (imgsz parameter)
- Consider model quantization for deployment

## Future Improvements

- [ ] Add validation dataset split
- [ ] Implement data augmentation
- [ ] Add model evaluation metrics
- [ ] Create web interface for predictions
- [ ] Add real-time webcam detection
- [ ] Export model for mobile deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational purposes. Please ensure compliance with dataset licensing terms.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection framework
- [CVAT.ai](https://cvat.ai) for annotation tools
- Alpaca dataset contributors

## Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This project is part of a deep learning portfolio focusing on computer vision applications.
