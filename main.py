import pandas as pd
import torch
import numpy as np
import os
from torchvision import transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import time
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

from train import train
from evaluation import evaluate
from dataset import GTSRB
from models import CustomResNet50

# Before starting, clear the memory
torch.cuda.empty_cache()

# Defining hyperparameters
batch_size = 256
learning_rate = 0.001
epochs = 25
num_classes = 43

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/kaggle/input/gtsrb-german-traffic-sign"

mean = torch.tensor([0.3403, 0.3122, 0.3215])
std = torch.tensor([0.1631, 0.1630, 0.1727])
# Create Transforms
# Define your transformations for data augmentation
train_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),  # Resizing the image to 64x64
        transforms.RandomHorizontalFlip(),  # Randomly flipping the image horizontally
        transforms.RandomRotation(
            10
        ),  # Randomly rotating the image by up to 10 degrees
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),  # Randomly altering the image's brightness, contrast, saturation, and hue
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=mean, std=std),  # Normalizing the tensor
    ]
)

# For validation and testing, you usually apply only minimal transformations (e.g., resizing and normalization).
test_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),  # Resizing the image to 64x64
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=mean, std=std),  # Normalizing the tensor
    ]
)

# Create Datasets
train_val_dataset = GTSRB(root_dir=root_dir, train=True, transform=train_transforms)
testset = GTSRB(root_dir=root_dir, train=False, transform=test_transforms)

# Define the sizes of the training and validation subsets
train_size = int(0.8 * len(train_val_dataset))  # 80% of the dataset for training
val_size = len(train_val_dataset) - train_size  # remaining 20% for validation

# Split the dataset into training and validation subsets
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

# Create DataLoader instances for training and validation subsets
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

# Load Datasets
test_loader = DataLoader(testset, batch_size, shuffle=False, num_workers=2)

# Initialize the model
weights_path = (
    "/kaggle/input/resnet50/resnet50_imagenet_v1.pth"  # Adjust the path as necessary
)

model = CustomResNet50(num_classes, weights_path=weights_path)

# Define optimizer and criterion functions
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize the scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

criterion = nn.CrossEntropyLoss()

# If CUDA is available, convert model and loss to cuda variables
model.to(device)
criterion.to(device)

# Perform training
# List to save training and val loss and accuracies
train_loss_list = [0] * epochs
train_acc_list = [0] * epochs
val_loss_list = [0] * epochs
val_acc_list = [0] * epochs

best_val_loss = float("inf")


for epoch in range(epochs):
    print("Epoch-%d: " % (epoch))

    train_start_time = time.monotonic()
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    train_end_time = time.monotonic()

    val_start_time = time.monotonic()
    val_loss, val_acc = evaluate(model, val_loader, optimizer, criterion, device)
    val_end_time = time.monotonic()

    train_loss_list[epoch] = train_loss
    train_acc_list[epoch] = train_acc
    val_loss_list[epoch] = val_loss
    val_acc_list[epoch] = val_acc

    print(
        "Training: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds"
        % (train_loss, train_acc, train_end_time - train_start_time)
    )
    print(
        "Validation: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds"
        % (val_loss, val_acc, val_end_time - val_start_time)
    )
    print("")

    scheduler.step()

    # Check if the current validation loss is the best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save the model if it has the best validation loss so far
        best_model_path = "./best_pytorch_classification_resnet50.pth"
        torch.save(model.state_dict(), best_model_path)
        print(
            f"Epoch {epoch}: New best model saved at {best_model_path} with validation loss {val_loss:.4f}"
        )

# Plot loss and accuracies for training and validation data
_, axs = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
axs[0].plot(train_loss_list, label="train")
axs[0].plot(val_loss_list, label="val")
axs[0].set_title("Plot - Loss")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
legend = axs[0].legend(loc="upper right", shadow=False)

# Accuracy plot
axs[1].plot(train_acc_list, label="train")
axs[1].plot(val_acc_list, label="val")
axs[1].set_title("Plot - Accuracy")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy")
legend = axs[1].legend(loc="center right", shadow=True)


# evaluate for the testdata
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "/kaggle/input/gtsrb-german-traffic-sign"

batch_size = 256

# Create Transforms
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
    ]
)

num_classes = 43
testset = GTSRB(root_dir=root_dir, train=False, transform=transform)
test_loader = DataLoader(testset, batch_size, shuffle=False, num_workers=2)

# print(os.listdir('/kaggle/input/gtsrb-german-traffic-sign'))
Model_path = best_model_path
model = CustomResNet50(num_classes)
model.load_state_dict(torch.load(Model_path))
model = model.to(device)


# Generating labels of classes
num_classes = 43

num = range(num_classes)
labels = []
for i in num:
    labels.append(str(i))
labels = sorted(labels)
for i in num:
    labels[i] = int(labels[i])
print("List of labels : ")
print("Actual labels \t--> Class in PyTorch")
for i in num:
    print("\t%d \t--> \t%d" % (labels[i], i))

# Read the image labels from the csv file
# Note: The labels provided are all numbers, whereas the labels assigned by PyTorch dataloader are strings

df = pd.read_csv("/kaggle/input/gtsrb-german-traffic-sign/Test.csv")
numExamples = len(df)
labels_list = list(df.ClassId)

corr_classified = 0
total_images = 0

with torch.no_grad():
    model.eval()

    for images, labels in test_loader:  # Assuming test_loader has the correct labels
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total_images += labels.size(0)
        corr_classified += (predicted == labels).sum().item()

print("Number of correctly classified images = %d" % corr_classified)
print("Number of incorrectly classified images = %d" % (total_images - corr_classified))
print("Final accuracy = %f" % (corr_classified / total_images))
