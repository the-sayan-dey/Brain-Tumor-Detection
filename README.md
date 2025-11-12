# Brain Tumor Detection using Deep Learning

A deep learning-based image classification system for detecting and classifying brain tumors using Transfer Learning with VGG16 architecture. Achieves **97% test accuracy** across 4 tumor classes.

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

## 🎯 Overview

This project implements a Convolutional Neural Network (CNN) for automated brain tumor detection and classification from medical imaging data. The model leverages transfer learning with VGG16 pre-trained on ImageNet to achieve **97% test accuracy** and **98.81% training accuracy** across 4 tumor classes.

**Key Features:**
- Transfer learning with VGG16 architecture
- **97% overall accuracy** on test dataset
- Custom data augmentation pipeline
- Fine-tuning strategy for domain adaptation
- Memory-efficient batch data generator
- Multi-class tumor classification (4 classes)
- **5,700 training images** and **1,311 test images**

## 🏗️ Architecture

### Model Components

#### 1. Base Model: VGG16
- **Pre-trained weights:** ImageNet
- **Input shape:** 128×128×3 (RGB)
- **Configuration:** Excluding top classification layers
- **Total parameters:** ~14.7M in base network

#### 2. Transfer Learning Strategy
- **Frozen layers:** All layers except last 3 convolutional layers
- **Fine-tuned layers:** `block5_conv1`, `block5_conv2`, `block5_conv3`
- **Rationale:** Balance between feature extraction and task-specific learning

#### 3. Custom Classification Head
```
Input (128×128×3)
    ↓
VGG16 Base Model (Frozen + Fine-tuned)
    ↓
Flatten Layer
    ↓
Dropout (0.3)
    ↓
Dense (128 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense (N classes, Softmax)
```

### Layer Details
| Layer Type | Output Shape | Parameters | Purpose |
|------------|--------------|------------|---------|
| VGG16 Base | (4, 4, 512) | 14,714,688 | Feature extraction |
| Flatten | (8192,) | 0 | Vectorization |
| Dropout | (8192,) | 0 | Regularization (30%) |
| Dense | (128,) | 1,048,704 | Feature compression |
| Dropout | (128,) | 0 | Regularization (20%) |
| Dense (Output) | (N,) | 129×N | Classification |

## 📁 Dataset Structure

```
Dataset/
├── Training/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class_2/
│   │   └── ...
│   └── class_N/
│       └── ...
└── Testing/
    ├── class_1/
    ├── class_2/
    └── class_N/
```

## 🔧 Installation

### Prerequisites
```bash
Python 3.7+
TensorFlow 2.x
Keras
NumPy
Pillow
Matplotlib
scikit-learn
```

### Install Dependencies
```bash
pip install tensorflow numpy pillow matplotlib scikit-learn
```

## 🚀 Usage

### 1. Prepare Your Dataset
Place your training and testing images in the appropriate directory structure as shown above.

### 2. Train the Model
```python
# Run the training cells in sequence
python train.py
```

### 3. Visualize Training Data
```python
# Display random samples from training set
python visualize_data.py
```

### 4. Evaluate Model
```python
# Generate training history plots
python plot_results.py
```

## 📊 Model Performance

### Training Configuration
- **Optimizer:** Adam (learning rate: 0.0001)
- **Loss Function:** Sparse Categorical Crossentropy
- **Batch Size:** 20
- **Epochs:** 10
- **Image Resolution:** 128×128 pixels
- **Total Training Samples:** 5,700 images
- **Steps per Epoch:** 285

### Training Results

#### Epoch-wise Performance
| Epoch | Loss | Accuracy | Training Time |
|-------|------|----------|---------------|
| 1/10 | 0.4596 | 82.46% | 307s |
| 2/10 | 0.2356 | 90.93% | 310s |
| 3/10 | 0.1694 | 93.43% | 321s |
| 4/10 | 0.1161 | 95.43% | 316s |
| 5/10 | 0.0837 | 96.82% | 469s |
| 6/10 | 0.0631 | 97.77% | 518s |
| 7/10 | 0.0601 | 97.65% | 296s |
| 8/10 | 0.0596 | 98.05% | 454s |
| 9/10 | 0.0526 | 98.09% | 488s |
| 10/10 | 0.0343 | **98.81%** | 361s |

**Final Training Accuracy:** 98.81%  
**Final Training Loss:** 0.0343

### Test Set Performance

#### Overall Metrics
- **Test Accuracy:** 97%
- **Total Test Samples:** 1,311 images

#### Class-wise Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Class 0** | 0.98 | 0.93 | 0.95 | 300 |
| **Class 1** | 0.93 | 0.96 | 0.94 | 306 |
| **Class 2** | 0.99 | 1.00 | 1.00 | 405 |
| **Class 3** | 0.99 | 0.99 | 0.99 | 300 |

#### Aggregate Metrics
- **Macro Average:** Precision: 0.97 | Recall: 0.97 | F1-Score: 0.97
- **Weighted Average:** Precision: 0.97 | Recall: 0.97 | F1-Score: 0.97

### Data Augmentation
- **Brightness adjustment:** Random factor (0.8-1.2×)
- **Contrast adjustment:** Random factor (0.8-1.2×)
- **Normalization:** Pixel values scaled to [0, 1]

## 🔬 Technical Details

### Transfer Learning Approach
The model employs a sophisticated transfer learning strategy:
1. **Feature Extraction:** Lower VGG16 layers (frozen) capture generic visual features
2. **Fine-tuning:** Top 3 convolutional layers adapt to medical imaging characteristics
3. **Custom Head:** Specialized classification layers for tumor detection

### Regularization Techniques
- **Dropout layers:** 30% and 20% rates prevent overfitting
- **Data augmentation:** Increases training data diversity
- **Batch normalization:** Inherited from VGG16 architecture

### Memory Optimization
- **Custom data generator:** Loads images in batches to manage memory efficiently
- **On-the-fly augmentation:** Reduces storage requirements
- **Dynamic batch processing:** Handles large datasets without RAM overflow

## 📈 Results

### Key Achievements
✅ **98.81% Training Accuracy** - Exceptional learning capability  
✅ **97% Test Accuracy** - Strong generalization to unseen data  
✅ **Class 2 Perfect Score** - 100% recall and 99% precision  
✅ **Balanced Performance** - All classes achieve >93% metrics  
✅ **Low Final Loss** - 0.0343 indicates excellent convergence  

### Training Insights
- **Rapid Learning:** Accuracy jumped from 82.46% to 90.93% in just 2 epochs
- **Steady Improvement:** Consistent accuracy gains throughout training
- **Minimal Overfitting:** Test accuracy (97%) close to training accuracy (98.81%)
- **Robust Classification:** All tumor classes identified with high precision and recall

## 🎓 Key Learnings

1. **Transfer Learning Efficiency:** Pre-trained VGG16 significantly reduces training time
2. **Medical Imaging Adaptation:** Fine-tuning top layers adapts generic features to medical domain
3. **Regularization Importance:** Dropout crucial for small medical datasets
4. **Augmentation Impact:** Brightness/contrast variations improve model robustness

## 🔮 Future Improvements

- [ ] Implement cross-validation for robust evaluation
- [ ] Add confusion matrix and classification report
- [ ] Experiment with other architectures (ResNet, EfficientNet)
- [ ] Implement Grad-CAM for model interpretability
- [ ] Add test set evaluation metrics
- [ ] Deploy model as web application
- [ ] Implement ensemble methods
- [ ] Add real-time inference pipeline

## 📚 References

- **VGG16 Paper:** [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- **Transfer Learning:** [A Survey on Transfer Learning](https://ieeexplore.ieee.org/document/5288526)
- **Medical Image Analysis:** Domain-specific deep learning applications

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

**Your Name**  
Email: your.email@example.com  
LinkedIn: [Your LinkedIn Profile]  
GitHub: [@yourusername](https://github.com/yourusername)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- VGG16 architecture by Visual Geometry Group, Oxford
- ImageNet dataset for pre-trained weights
- TensorFlow and Keras development teams

---

**⭐ If you found this project helpful, please consider giving it a star!**
