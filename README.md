# Tumour Segmentation in Medical Images

### Image Processing & Machine Vision

This project investigates **brain tumour segmentation from medical MRI images** using image processing, machine vision, and deep learning techniques. The project uses the **BraTS (Brain Tumour Segmentation Challenge)** dataset and explores the complete segmentation pipeline, including data preparation, preprocessing, augmentation, model training, optimisation, evaluation, and inference.

The main objective is to automatically identify tumour regions from MRI images at the pixel level and evaluate the effectiveness of the segmentation model using standard medical image segmentation metrics such as the **Dice Coefficient** and **Intersection over Union (IoU)**.

---

## 1. Project Overview

Medical image segmentation is an important application of machine vision and deep learning. Unlike image classification, which predicts a label for an entire image, segmentation assigns a class label to individual pixels.

For brain tumour analysis, the goal is to distinguish tumour regions from normal brain tissue based on MRI scans.

The general workflow of this project is:

**Medical MRI Images → Preprocessing → Data Augmentation → Segmentation Model → Training & Optimisation → Predicted Tumour Mask → Evaluation**

The project focuses on the following tasks:

* Preparing and processing medical imaging data
* Performing pixel-level tumour segmentation
* Applying data augmentation to improve dataset diversity
* Training a deep learning segmentation model
* Addressing class imbalance between tumour and background pixels
* Evaluating segmentation accuracy using Dice and IoU
* Analysing model limitations and possible improvements

---

## 2. Dataset

### 2.1 Data Source

The dataset used in this project is mainly obtained from the **BraTS (Brain Tumour Segmentation Challenge)**.

BraTS provides MRI brain scans together with expert-annotated tumour regions and is widely used for benchmarking brain tumour segmentation methods.

The annotations provide ground-truth information indicating tumour regions, allowing the segmentation problem to be treated as a supervised learning task.

In a segmentation task, labels are defined at the **pixel level**, rather than assigning only one label to the complete image. Therefore, the model learns to predict which pixels belong to tumour regions.

---

### 2.2 Data Generation and Annotation

The medical images used for training contain corresponding tumour annotations.

These annotations act as the **ground-truth segmentation masks** during model training and evaluation.

For supervised segmentation, accurate and consistent annotation is particularly important because incorrect pixel labels directly affect the features learned by the model.

In general medical image segmentation projects, annotations may be generated using tools such as:

* LabelMe
* VGG Image Annotator
* Other specialised medical image annotation software

For this project, existing annotations provided by the BraTS dataset are used rather than manually creating tumour boundaries.

---

## 3. Data Preprocessing and Augmentation

Medical images may contain differences in intensity distributions, orientation, resolution, and noise. Preprocessing is therefore an important step before model training.

The main preprocessing and augmentation operations implemented in this project are:

### Normalization

Pixel values are normalised to a standard range or distribution.

Normalization improves consistency between images and helps make neural network optimisation more stable.

### Horizontal Flipping

Images and their corresponding segmentation masks are horizontally flipped.

This increases the diversity of the training samples while preserving the spatial relationship between an MRI image and its tumour annotation.

### Random Rotation

Images are randomly rotated by a selected angle.

Rotation augmentation allows the model to observe structures at different orientations and reduces its dependence on a particular image orientation.

### Other Potential Preprocessing Techniques

Additional image processing techniques that can be considered include:

* **Rescaling / Resizing** — converting images to a uniform input size
* **Gaussian smoothing** — reducing image noise
* **Median filtering** — reducing impulse noise while preserving edges
* **Contrast enhancement** — increasing the visibility of anatomical structures
* **Cropping** — removing unnecessary background regions

These techniques are useful theoretical extensions but were not necessarily all applied in the final implementation.

---

## 4. Final Dataset

Considering the computational capability of the available laptop and GPU, the final dataset was divided into training, validation, and testing sets.

| Dataset    | Number of Images | Percentage |
| ---------- | ---------------: | ---------: |
| Training   |           10,850 |        70% |
| Validation |            2,325 |        15% |
| Testing    |            2,325 |        15% |
| **Total**  |       **15,500** |   **100%** |

The **training set** is used to update model parameters.

The **validation set** is used to monitor model performance and support hyperparameter tuning during development.

The **testing set** is kept separate from training and is used to evaluate the final model's generalisation performance.

Maintaining independent training, validation, and testing datasets is important for preventing **data leakage**.

---

## 5. Segmentation Approaches

Tumour segmentation can be performed using both traditional image processing techniques and deep learning methods.

### 5.1 Classical Image Processing

#### Thresholding

Thresholding separates regions according to pixel intensity.

Although computationally efficient, simple thresholding is often insufficient for MRI tumour segmentation because tumour and normal tissue intensities can overlap.

#### Region Growing

Region growing begins from selected seed pixels and expands a region according to intensity and connectivity criteria.

It can produce connected segmentation regions but is sensitive to noise, seed selection, and intensity variation.

---

## 6. Deep Learning for Medical Image Segmentation

Deep learning methods can automatically learn hierarchical features from medical images and are generally more suitable for complex segmentation problems.

### Convolutional Neural Networks

Convolutional Neural Networks (CNNs) extract spatial features using convolutional operations.

Important CNN parameters include:

* Kernel size
* Padding
* Number of filters
* Activation functions
* Batch size
* Learning rate

### U-Net

**U-Net** is a widely used architecture for biomedical image segmentation.

Its encoder-decoder structure allows the network to learn high-level semantic features while recovering spatial information required for pixel-level predictions. Skip connections can transfer spatial information from encoder layers to corresponding decoder layers.

### Other Possible Approaches

More advanced approaches that may be investigated in future development include:

* DeepLab
* ResNet-based segmentation models
* DenseNet-based models
* Attention mechanisms
* Multimodal MRI fusion
* Transfer learning and pretrained networks

These approaches represent possible extensions rather than all being part of the current implementation.

---

## 7. Model Training

The model was trained using **GPU acceleration with CUDA** where available.

The main training configuration included:

| Parameter             | Configuration       |
| --------------------- | ------------------- |
| Epochs                | 10                  |
| Optimizer             | Adam                |
| Initial Loss          | Cross-Entropy Loss  |
| Activation            | ReLU                |
| Normalization         | Batch Normalization |
| Hardware Acceleration | CUDA / GPU          |

Other important hyperparameters include:

* Batch size
* Learning rate
* Kernel size
* Padding
* Number of convolutional filters

The Adam optimiser is used to update network parameters by minimising the training loss.

ReLU introduces non-linearity into the network, while Batch Normalization can help stabilise the distribution of intermediate activations during training.

---

## 8. Class Imbalance and Loss Function Optimisation

A major challenge in tumour segmentation is **class imbalance**.

In an MRI image, normal/background pixels usually occupy a much larger area than tumour pixels. Consequently, a model can achieve apparently reasonable pixel accuracy while still failing to correctly segment the relatively small tumour region.

To address this problem, class weighting was introduced according to the frequency of different classes.

The initial implementation used:

**Cross-Entropy Loss**

During optimisation, the following loss functions were considered or applied:

### Weighted Cross-Entropy Loss

Weighted Cross-Entropy assigns greater importance to underrepresented classes.

This reduces the dominance of background pixels during optimisation.

### Dice Loss

Dice Loss directly considers the overlap between the predicted tumour region and ground-truth segmentation mask.

It is particularly useful for segmentation problems with imbalanced foreground and background regions.

Combining or comparing **Weighted Cross-Entropy Loss** and **Dice Loss** can therefore improve the model's sensitivity to tumour regions.

---

## 9. Model Evaluation

The model is evaluated using both segmentation accuracy and computational performance.

The main evaluation measurements include:

* Memory usage per batch
* Throughput per batch
* Average Dice Coefficient
* Average Intersection over Union (IoU)

### Dice Coefficient

The Dice Coefficient measures the similarity between the predicted segmentation and the ground-truth mask:

$$
Dice = \frac{2|P \cap G|}{|P| + |G|}
$$

where:

* \(P\) is the predicted tumour region
* \(G\) is the ground-truth tumour region

A Dice score closer to **1** indicates greater overlap and therefore better segmentation performance.

### Intersection over Union (IoU)

IoU, also known as the Jaccard Index, measures the intersection between the prediction and ground truth relative to their union:

$$
IoU = \frac{|P \cap G|}{|P \cup G|}
$$

An IoU value closer to **1** represents better segmentation.

### Additional Evaluation Metrics

Other metrics that could be incorporated into future experiments include:

* Precision
* Recall / Sensitivity
* Specificity
* F1-score
* Pixel Accuracy

For medical tumour detection, recall can be particularly informative because it measures how many actual positive tumour pixels are successfully detected.

---

## 10. Inference Results and Analysis

The experimental results indicate that the current model still has substantial room for improvement.

The training and validation losses suggest an **underfitting trend**, while the average Dice Coefficient and IoU remain relatively low.

This indicates that the current model has not yet learned sufficiently discriminative representations for accurate tumour segmentation.

Several factors may contribute to the current performance:

* Limited number of training epochs
* Suboptimal learning rate or batch size
* Class imbalance between tumour and non-tumour pixels
* Insufficient model capacity
* Limited data augmentation
* Incomplete hyperparameter optimisation
* Complexity and variability of tumour appearance

Therefore, the current implementation should be considered a baseline segmentation system rather than a fully optimised medical segmentation model.

---

## 11. Potential Improvements

Future development can focus on improving both segmentation accuracy and model generalisation.

Possible improvements include:

1. **Increase the number of training epochs**
   Ten epochs may be insufficient for the network to fully converge.

2. **Hyperparameter tuning**
   Experiment with different learning rates, batch sizes, kernel sizes, padding strategies, and network depths.

3. **Improve the loss function**
   Test combinations such as Dice Loss with Weighted Cross-Entropy Loss or other segmentation-oriented loss functions.

4. **Expand data augmentation**
   Introduce random cropping, scaling, brightness adjustment, elastic transformation, and other medically appropriate augmentations.

5. **Use specialised segmentation architectures**
   Experiment with U-Net variants, DeepLab, attention-based networks, or other segmentation models.

6. **Use multimodal MRI information**
   Different MRI modalities contain complementary information. Combining them may improve tumour boundary detection.

7. **Regularisation**
   Techniques such as dropout and weight decay can be investigated to improve generalisation when model complexity increases.

8. **Detailed error analysis**
   Poor segmentation cases should be analysed individually to determine whether errors originate from tumour size, tumour location, image quality, preprocessing, or model limitations.

---

## 12. Dataset Quality Considerations

Dataset quality has a direct influence on segmentation performance.

Important considerations include:

### Diversity

The dataset should contain sufficient variation in tumour:

* Size
* Shape
* Location
* Intensity
* Patient anatomy

### Annotation Accuracy

Ground-truth masks should accurately represent tumour regions. Inconsistent labels may negatively affect supervised learning.

### Dataset Bias

The dataset should represent different imaging conditions and patient cases as broadly as possible to reduce bias and improve generalisation.

### Data Leakage

Images or closely related samples from the same source should not unintentionally appear across training and testing subsets in a way that gives an unrealistic estimate of model performance.

---

## 13. Equipment and Resources

### Hardware

* Laptop / High-performance computer
* CUDA-compatible GPU

### Operating System

* Windows

### Programming Language

* Python

### Development Environment

* PyCharm

### Main Python Libraries

* PyTorch
* TensorFlow
* OpenCV
* NumPy
* Matplotlib
* SimpleITK
* ITK

### Dataset and Research Resources

* BraTS Brain Tumour Segmentation Challenge dataset
* Medical image processing literature
* Machine vision and deep learning research papers

---

## 14. Workflow

```text
BraTS MRI Dataset
       │
       ▼
Data Preparation & Annotation
       │
       ▼
Preprocessing
Normalization / Resizing
       │
       ▼
Data Augmentation
Flipping / Random Rotation
       │
       ▼
Train / Validation / Test Split
70% / 15% / 15%
       │
       ▼
Segmentation Model
       │
       ▼
Training with CUDA / GPU
Adam Optimizer
       │
       ▼
Loss Optimisation
Cross-Entropy
Weighted Cross-Entropy / Dice Loss
       │
       ▼
Model Inference
       │
       ▼
Predicted Tumour Masks
       │
       ▼
Evaluation
Dice / IoU / Memory / Throughput
```

---

## 15. Conclusion

This project demonstrates the development of a machine vision pipeline for **brain tumour segmentation from MRI images**.

The BraTS dataset provides medical images and corresponding tumour annotations for supervised segmentation. The dataset is processed using normalization and data augmentation techniques such as horizontal flipping and random rotation before being divided into training, validation, and testing subsets.

A deep learning segmentation model is trained using GPU acceleration, with Adam used for optimisation. Cross-Entropy Loss is initially applied, while Dice Loss and Weighted Cross-Entropy Loss are investigated to address the significant class imbalance between tumour and background pixels.

The current experimental results show relatively low Dice and IoU scores and indicate underfitting. Therefore, further optimisation of the network architecture, training duration, hyperparameters, loss functions, and augmentation strategies is required.

Despite these limitations, the project establishes a complete baseline workflow from **medical image preprocessing and model training to segmentation inference and quantitative evaluation**, providing a foundation for further research and development in automated brain tumour segmentation.

---

## Disclaimer

This project is developed for **educational and research purposes only**.

The segmentation results produced by this project are not intended for clinical diagnosis, treatment planning, or other medical decision-making.
