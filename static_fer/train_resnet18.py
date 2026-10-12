import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models # Moedls is for pretrained models
from torch.utils.data import DataLoader

from tqdm import tqdm

# ------------------
# Device
# ------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# ------------------
# Transform
# ------------------

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # Convert to 3 channels
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406], # ImageNet mean
        std = [0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # Convert to 3 channels
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406], # ImageNet mean
        std = [0.229, 0.224, 0.225]
    )
])

# ------------------
# Dataset and DataLoader
# ------------------

train_dataset = datasets.ImageFolder(
    root = "data_ckplus_static_split/train",
    transform = train_transform
)

test_dataset = datasets.ImageFolder(
    root = "data_ckplus_static_split/test",
    transform = test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size = 32,
    shuffle = True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 32,
    shuffle = False,
)

print("Classes:", train_dataset.classes)

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))

# ------------------
# ResNet18 Model
# ------------------

model = models.resnet18(
    weights = models.ResNet18_Weights.DEFAULT
)

num_features = model.fc.in_features
model.fc = nn.Linear(
    num_features,
    len(train_dataset.classes)
)

model = model.to(device)

# ------------------
# Loss and Optimizer
# ------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr = 0.001
)

# ------------------
# Train function
# ------------------

def train_one_epoch(model, loader):
    model.train() # Set model to training mode
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

# ------------------
# Evaluation function
# ------------------

def evaluate(model, loader):
    model.eval() # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

# ------------------
# Main training loop
# ------------------

num_epochs = 10
best_acc = 0.0

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader
    )

    print(
        f"Train Loss: {train_loss:.4f},"
        f"Train Acc: {train_acc:.4f} | "
        f"Test Loss: {test_loss:.4f},"
        f"Test Acc: {test_acc:.4f}"
    )

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 
        "resnet18_static_fer.pth")
        print("Best model saved.")

print("Training finished.")
print("Best of Acc:", best_acc)