import pandas as pd
import dataset
import torch
import os
from models import CustomResNet50
from evaluation import evaluate
import torchvision
from torchvision import transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
import time
import numpy as np
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

from train import train

# Before starting, clear the memory
torch.cuda.empty_cache()

# Defining hyperparameters
batch_size = 256
learning_rate = 0.001
epochs = 15
num_classes = 43

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


root_dir = "../Dataset"

# Create Transforms
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
    ]
)

# Create Datasets
train_val_dataset = dataset.GTSRB(root_dir=root_dir, train=True, transform=transform)
testset = dataset.GTSRB(root_dir=root_dir, train=False, transform=transform)

# Define the sizes of the training and validation subsets
train_size = int(0.8 * len(train_val_dataset))  # 80% of the dataset for training
val_size = len(train_val_dataset) - train_size  # remaining 20% for validation

# Split the dataset into training and validation subsets
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

# Create DataLoader instances for training and validation subsets
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

# Load Datasets
testloader = DataLoader(testset, batch_size, shuffle=False, num_workers=2)

# Initialize the model
model = CustomResNet50(num_classes)

# Define optimizer and criterion functions
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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

for epoch in range(epochs):
    print("Epoch-%d: " % (epoch))

    train_start_time = time.monotonic()
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    train_end_time = time.monotonic()

    val_start_time = time.monotonic()
    val_loss, val_acc = evaluate(model, val_loader, optimizer, criterion)
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


# Saving the model

# Create folder to save model
Model_folder = "./Models"
if not os.path.isdir(Model_folder):
    os.mkdir(Model_folder)

PATH_TO_MODEL = Model_folder + "/pytorch_classification_resnet50.pth"
if os.path.exists(PATH_TO_MODEL):
    os.remove(PATH_TO_MODEL)
torch.save(model.state_dict(), PATH_TO_MODEL)

print("Model saved at %s" % (PATH_TO_MODEL))


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
