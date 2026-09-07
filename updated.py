import numpy as np
import pandas as pd
# import tarfile
import glob
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from torchvision.transforms import functional
import random
from torch.utils.data import Subset
import scipy


# print("Loading image...")
# Loading the dataset
# zip_file = tarfile.open("BraTS2021_Training_Data.tar")

# print("Extracting...")
# zip_file.extractall("data")
# zip_file.close()


# BraTSDataset
print("Creating BraTSDataset...")


class BraTSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.patients = os.listdir(root_dir)[
                        :10]  # limiting the loading of dataset to reducing training time as Kaggle have limited GPU

        self.patients = [p for p in os.listdir(root_dir) if not p.startswith('.')]  # 过滤掉隐藏文件

        self.num_slices_per_patient = 155
        self.index_mapping = [
            (patient_id, slice_idx)
            for patient_id in self.patients
            for slice_idx in range(self.num_slices_per_patient)
        ]

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        patient_id, slice_idx = self.index_mapping[idx]
        patient_path = os.path.join(self.root_dir, patient_id)

        # Loading modalities
        t1ce_path = os.path.join(patient_path, f"{patient_id}_t1ce.nii.gz")
        t1ce_img = nib.load(t1ce_path).get_fdata()
        t1ce_slice = t1ce_img[:, :, slice_idx]
        t1ce_slice = t1ce_slice[np.newaxis, ...]  # to create a placeholder for n_channel dimension for model requirement

        # Loading segmentation mask
        seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
        seg_img = nib.load(seg_path).get_fdata()
        seg_slice = seg_img[:, :, slice_idx]

        sample = (t1ce_slice, seg_slice)

        if self.transform:
            sample = self.transform(sample)

        return t1ce_slice, seg_slice


# Data Preprocessing and Augmentation
print("Data Preprocessing and Augmentation...")


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation."""

    def __call__(self, sample):
        image, mask = sample
        epsilon = 1e-6
        image = (image - image.mean()) / (image.std() + epsilon)
        return image, mask


class HorizontalFlip(object):
    """Apply horizontal flipping of an image and its mask."""

    def __call__(self, sample):
        image, mask = sample
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
        return image, mask


class RandomRotation(object):
    """Randomly rotate an image and its mask."""

    def __call__(self, sample):
        image, mask = sample
        angle = random.randint(0, 360)  # You can adjust the range of angles
        image = scipy.ndimage.rotate(image, angle, reshape=False)
        mask = scipy.ndimage.rotate(mask, angle, reshape=False)
        return image, mask


all_transformation = transforms.Compose([Normalize(), HorizontalFlip(), RandomRotation()])


# Sample Visualization
# Initialize the dataset
print("Initialize the dataset...")

brats_dataset = BraTSDataset(root_dir='data', transform=all_transformation)

# Visualize one sample - T1ce Modality with overlaid segmentation mask
print("Sample Visualization...")

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(brats_dataset[100][0][0, :, :], cmap='gray')
ax.imshow(brats_dataset[100][1], cmap='gray', alpha=0.5)  # alpha controls the transparency

ax.set_title('T1ce Modality with Segmentation Mask Overlay')
ax.axis('off')

plt.show()


# Dataset Splitting and DataLoader Creation
# Split the dataset into train(70%), val(15%) and test(15%)
print("Dataset Splitting and DataLoader Creation...")

dataset_size = len(brats_dataset)
indices = list(range(dataset_size))
split_train = int(np.floor(0.7 * dataset_size))
split_val = int(np.floor(0.85 * dataset_size))
np.random.shuffle(indices)


# Creating data loader for training, validation, and test splits
print("Creating data loader for training, validation, and test splits...")

train_indices, val_indices, test_indices = indices[:split_train], indices[split_train:split_val], indices[split_val:]

train_dataset = Subset(brats_dataset, train_indices)
val_dataset = Subset(brats_dataset, val_indices)
test_dataset = Subset(brats_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=32, shuffle=False)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")


# Create 2D U-Net model
print("Creating 2D U-Net model...")


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU => Dropout) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_prob=0.5):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout_prob),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_prob=0.5):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# Model Initialization, Loss Function, and Optimizer
print("Model Initialization, Loss Function, and Optimizer...")


# Dice Loss function
def dice_loss(pred, target, smooth=1e-6):
    """Calculates the Dice Loss."""
    # Convert logits to predicted class
    pred = torch.argmax(pred, dim=1)  # Predicted class (shape [batch_size, height, width])

    # Ensure target is of the correct shape (it should be [batch_size, height, width])
    target = target.squeeze(1)  # Remove singleton dimension

    # Calculate the intersection and union
    intersection = torch.sum(pred * target)
    dice = (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
    return 1 - dice


# Combined loss function: Cross Entropy + Dice Loss
def combined_loss(pred, target, alpha=0.5):
    """Combines Cross Entropy Loss and Dice Loss."""
    # Cross entropy loss (no need for softmax as CrossEntropyLoss applies it internally)
    ce_loss = F.cross_entropy(pred, target)

    # Dice loss (use the predicted logits, no need to apply softmax here)
    dice = dice_loss(pred, target)

    # Return the weighted sum of cross-entropy and dice loss
    return alpha * ce_loss + (1 - alpha) * dice


def weighted_cross_entropy_loss(pred, target, weights=None):
    if weights is not None:
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion(pred, target)


# Calculate class weights
print("Calculate class weights...")


def calculate_class_weights(dataset):
    # Count class occurrences in the dataset
    class_counts = [0] * 4  # Assuming 4 classes
    for _, target in dataset:
        for i in range(4):  # Assuming the classes are 0, 1, 2, 3
            class_counts[i] += (target == i).sum().item()

    # Calculate weights as the inverse of class frequencies
    total_pixels = sum(class_counts)
    class_weights = [total_pixels / c if c != 0 else 0 for c in class_counts]
    class_weights = torch.tensor(class_weights).float()

    return class_weights


# A U-Net model with one input channel(for grayscale) and four output classes is initialized to segment the MRI slices into four categories.
# multi-class segmentation: Each pixel in the image will have one of the four possible labels (background, tumor core, enhancing tumor, and edema).
# use a loss function like nn.CrossEntropyLoss() for multi-class classification, where each pixel is assigned to one of the 4 classes

class_weights = calculate_class_weights(brats_dataset)
model = UNet(n_channels=1, n_classes=4)
loss_fn = lambda pred, target: weighted_cross_entropy_loss(pred, target, weights=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # L2 regularization, learning rate


# Device Configuration for Training
if torch.cuda.is_available():
    print("CUDA is available. GPU will be used for training.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Training will be on CPU.")
    device = torch.device("cpu")


# Model Training and Validation Process
print("Model Training and Validation Process...")
# Original segmentation output are [0,1,2,4]. To make it compatible with the softmax the labels are adjusted.


def remap_labels(y):
    y[y == 4] = 3
    return y


num_epochs = 10
model = model.to(device)  # Ensure the model is on GPU
training_losses = []
validation_losses = []


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for X, y in train_loader:
        y = remap_labels(y).long()
        X, y = X.to(device).float(), y.to(device)  # Move data to GPU
        optimizer.zero_grad()

        # Forward pass, calculate loss
        print("Forward pass, calculate loss...")

        output_masks = model(X)
        #loss = loss_fn(output_masks, y)
        loss = combined_loss(output_masks, y)  # Use combined loss function
        train_loss += loss.item()

        # Backward pass and optimization
        print("Backward pass and optimization...")

        loss.backward()
        optimizer.step()

    # Calculate average training loss for the epoch
    print("Calculate average training loss for the epoch...")

    avg_train_loss = train_loss / len(train_loader)
    training_losses.append(avg_train_loss)

    # Validation step
    print("Validation step...")

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X, y in val_loader:
            y = remap_labels(y).long()
            X, y = X.to(device).float(), y.to(device)
            output_masks = model(X)
            #loss = loss_fn(output_masks, y)
            loss = combined_loss(output_masks, y)
            val_loss += loss.item()

    # Calculate average validation loss for the epoch
    print("Calculate average validation loss, average dice, average IoU for the epoch...")

    avg_val_loss = val_loss / len(val_loader)
    validation_losses.append(avg_val_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

    checkpoint_path = f'working_5/model_epoch_{epoch + 1}.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, checkpoint_path)


# Plotting the training and validation loss
print("Plotting the training and validation loss...")

# Fix by swapping if necessary
if len(training_losses) > 0 and len(validation_losses) > 0:
    if training_losses[-1] < validation_losses[-1]:
        # Swap the lists if the loss order is incorrect
        training_losses, validation_losses = validation_losses, training_losses

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), training_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# Model Testing Process
print("Model Testing Process...")


def test_image(model, image_idx, dataset):
    model.eval()
    image, true_mask = dataset[image_idx]
    image = torch.tensor(image).unsqueeze(0).to(device).float()  # Add batch dimension and move to device

    with torch.no_grad():
        prediction = model(image)
        prediction = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()  # Get the class with the max score

    return image.squeeze().cpu().numpy(), prediction

# Test a specific image (e.g., the 100th one)
print("Testing the model with a given image...")

test_img, test_pred = test_image(model, 100, brats_dataset)

# Visualize the result
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(test_img, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(test_pred, cmap='jet')
ax[1].set_title('Predicted Segmentation Mask')
ax[1].axis('off')

plt.show()


# Model Performance Evaluation: Memory Usage and Throughput
print("Model Performance Evaluation: Memory Usage and Throughput...")

import time


def evaluate_performance(model, data_loader, device):
    model.eval()
    memory_usages = []
    throughputs = []
    batch_sizes = []

    with torch.no_grad():
        for X, _ in data_loader:
            batch_sizes.append(X.size(0))

            # Measure memory usage
            initial_memory = torch.cuda.memory_allocated(device)
            X = X.to(device).float()

            # Measure throughput
            start_time = time.time()
            model(X)
            end_time = time.time()

            current_memory = torch.cuda.memory_allocated(device)
            memory_usage = current_memory - initial_memory
            memory_usages.append(memory_usage)

            # Calculate throughput for this batch
            throughput = X.size(0) / (end_time - start_time)
            throughputs.append(throughput)

    return memory_usages, throughputs, batch_sizes

# Evaluate the model
print("Evaluate the model...")

memory_usages, throughputs, batch_sizes = evaluate_performance(model, val_loader, device)

# Plotting the results
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Batch Number')
ax1.set_ylabel('Memory Usage (bytes)', color=color)
ax1.plot(memory_usages, color=color, marker='o', label='Memory Usage')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Throughput (images/sec)', color=color)
ax2.plot(throughputs, color=color, marker='x', label='Throughput')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Memory Usage and Throughput per Batch During Inference')
plt.show()


# Evaluation Metrics for Segmentation: Dice Coefficient and IoU Score
print("Evaluation Metrics for Segmentation: Dice Coefficient and IoU Score...")


def dice_coefficient(predicted, target):
    """
    Compute the Dice Coefficient.
    :param predicted: the logits or the prediction from the model.
    :param target: the target mask.
    :return: the Dice coefficient.
    """
    smooth = 1.0
    predicted = torch.sigmoid(predicted)
    predicted = (predicted > 0.5).float()

    intersection = (predicted * target).sum()
    dice = (2. * intersection + smooth) / (predicted.sum() + target.sum() + smooth)
    return dice


def iou_score(predicted, target):
    """
    Compute the Intersection over Union (IoU) score.
    :param predicted: the logits or the prediction from the model.
    :param target: the target mask.
    :return: the IoU score.
    """
    predicted = torch.sigmoid(predicted)
    predicted = (predicted > 0.5).float()

    intersection = (predicted * target).sum()
    union = predicted.sum() + target.sum() - intersection

    iou = (intersection + 1.0) / (union + 1.0)
    return iou


model.eval()
dice_scores = []
iou_scores = []

with torch.no_grad():
    for images, true_masks in val_loader:
        images = images.to(device).float()
        true_masks = true_masks.to(device).float()

        preds = model(images)
        for pred, true_mask in zip(preds, true_masks):
            pred = torch.sigmoid(pred).float()
            pred = (pred > 0.5).float()

            dice_scores.append(dice_coefficient(pred, true_mask).item())
            iou_scores.append(iou_score(pred, true_mask).item())

print(f"Average Dice Coefficient: {sum(dice_scores) / len(dice_scores)}")
print(f"Average IoU Score: {sum(iou_scores) / len(iou_scores)}")
