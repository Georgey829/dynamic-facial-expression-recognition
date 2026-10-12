import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

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
test_dataset = datasets.ImageFolder(
    root = "data/test",
    transform = test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 32,
    shuffle = False
)

class_names = test_dataset.classes
print("Class names:", class_names)
print("Test size:", len(test_dataset))

# ------------------
# Create ResNet18 model
# ------------------

model = models.resnet18(weights=None)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(device)

# ------------------
# Load the trained model weights
# ------------------

model.load_state_dict(torch.load("resnet18_fer.pth", map_location=device))
model.eval()

print("Model loaded successfully.")

# ------------------
# Collect predicitions
# ------------------

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ------------------
# Classification Report and Confusion Matrix
# ------------------

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# ------------------
# Confusion Matrix
# ------------------

print("\nConfusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)


np.save("all_labels.npy", np.array(all_labels))
np.save("all_preds.npy", np.array(all_preds))

print("Prediction results saved.")