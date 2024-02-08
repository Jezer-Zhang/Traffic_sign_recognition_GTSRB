# Function to perform training of the model
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, loader, opt, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    # Train the model
    model.train()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Training pass
        opt.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # Backpropagation
        loss.backward()

        # Calculate accuracy
        acc = calculate_accuracy(output, labels)

        # Optimizing weights
        opt.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)
