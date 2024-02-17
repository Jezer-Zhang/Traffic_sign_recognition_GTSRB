import torch
from train import calculate_accuracy

# Function to perform evaluation on the trained model


def evaluate(model, loader, opt, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    # Evaluate the model
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Run predictions
            output = model(images)
            loss = criterion(output, labels)

            # Calculate accuracy
            acc = calculate_accuracy(output, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)
