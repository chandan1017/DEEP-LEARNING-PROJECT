print("Running Image Classification Script...")

# full code begins here
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

# === STEP 1: Data Loading ===

data_dir = r"C:\Users\SRIHARIKA\Downloads\Data Science\Task 2-Image Classification\dataset\train"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Classes: {dataset.classes}")
print(f"Total Images: {len(dataset)}")

# === STEP 2: Define the CNN Model ===

class SimpleCNN(nn.Module):
    def __init__(self):  # ✅ CORRECT
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, len(dataset.classes))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 64, 64]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 32, 32]
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()

# === STEP 3: Loss and Optimizer ===

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === STEP 4: Train the Model ===

for epoch in range(5):
    running_loss = 0.0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/5], Loss: {running_loss / len(dataloader):.4f}")

# === STEP 5: Evaluate ===

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

# === STEP 6: Predict on Test Images ===

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

test_folder = r"C:\Users\SRIHARIKA\Downloads\Data Science\Task 2-Image Classification\test_images"

print("\nPredictions on test images:")
for filename in os.listdir(test_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(test_folder, filename)
        image = Image.open(path).convert('RGB')
        image_tensor = transform_test(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            _, pred = torch.max(output, 1)
            predicted_class = dataset.classes[pred.item()]
            print(f"{filename} → {predicted_class}")

        # Show the image with prediction
        plt.imshow(image)
        plt.title(f"{filename}: {predicted_class}")
        plt.axis('off')
        plt.show()

# === Save the model ===
torch.save(model.state_dict(), "cat_dog_model.pth")

# full code ends hereprint("Running Image Classification Script...")