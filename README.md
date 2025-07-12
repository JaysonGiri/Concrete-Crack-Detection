# Concrete Crack Detection using CNN

A deep learning project that automatically detects cracks in concrete structures using Convolutional Neural Networks (CNN).

## Project Overview

This project uses computer vision and deep learning to classify concrete images as either "cracked" or "not cracked". Instead of manual inspection, the trained model can automatically identify structural damage in concrete surfaces.

## Project Structure

### Script Breakdown

**Data Loading:**
- Load concrete images from Positive and Negative directories
- Organize 4,000 images (2,000 cracked, 2,000 non-cracked)

**Data Exploration:**
- Display dataset statistics and class distribution
- Visualize sample images from both categories

**Data Preprocessing:**
- Resize all images to 100x100 pixels for consistent input
- Normalize pixel values from 0-255 to 0-1 range
- Split data into 75% training and 25% testing

**Feature Engineering:**
- Image resizing using OpenCV
- Pixel normalization for neural network optimization
- Binary label encoding (0 = no crack, 1 = crack)

**Model Training:**
- Custom CNN with 4 convolutional blocks
- Batch normalization and dropout for regularization
- 5 epochs with Adam optimizer

**Evaluation:**
- Performance metrics: accuracy, precision, recall
- Confusion matrix visualization
- Classification report generation

## Prerequisites
- Python 3.7+
- Required libraries:
  - tensorflow
  - opencv-python
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - pillow
  - pandas

```
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn pillow pandas
```

## Technologies Used

- **Python** - Programming language
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing
- **NumPy** - Numerical computations
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-learn** - Machine learning utilities

## Dataset

- **Total Images**: 4,000
- **Positive (Crack)**: 2,000 images
- **Negative (No Crack)**: 2,000 images
- **Image Size**: Resized to 100x100 pixels
- **Format**: JPG

## Model Architecture

Custom CNN with the following structure:
- 4 Convolutional blocks (32, 64, 128, 256 filters)
- MaxPooling for dimension reduction
- Dropout layers to prevent overfitting
- Dense layers for final classification
- Sigmoid activation for binary output


## Model Performance
The model achieves:

- **Training Accuracy:** 99.5%
- **Test Accuracy:** 98.96%
- **Test Precision:** 98.00%
- **Test Recall:** 99.96%


## Key Features
- **Data Processing:** Automated image loading, resizing, and normalization pipeline 
- **Model:** Deep CNN optimized for binary image classification 
- **Evaluation:** Comprehensive metrics including precision, recall, and F1-score 
- **Visualization:** Sample images, confusion matrix, and performance charts

## Configuration
Key parameters in the Config class:

- **data_dir_path:** Path to image dataset
- **epochs:** Number of training iterations (5)
- **learning_rate:** Adam optimizer rate (default: 0.001)
- **dropout_rate:** Regularization strength (default: 0.5)

## Applications
- **Infrastructure Monitoring:** Automated bridge and building inspection
- **Preventive Maintenance:** Early crack detection for timely repairs
- **Quality Control:** Construction material assessment
- **Safety Assessment:** Structural health monitoring

## Future Enhancements
- Implement data augmentation for improved generalization
- Add transfer learning with pre-trained models
- Optimize hyperparameters using grid search
- Deploy model as web application or mobile app

## Author
Sanjay Giri Prakash - sanjaygirip@gmail.com

