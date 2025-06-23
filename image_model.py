import torch   
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader 
from google.colab import drive
import os 

# 1. Mount Google Drive to save model
drive.mount('/content/drive')

# 2. image transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),                  
    transforms.ToTensor(),                          # Convert images to PyTorch tensors (C x H x W)
    transforms.Normalize([0.5, 0.5, 0.5],           
                         [0.5, 0.5, 0.5])
])

# 3. dataset path
dataset_path = '/content/Dataset' 

# 4. Load the datasets using ImageFolder for Train, Validation, Test
train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'Train'), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'Validation'), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'Test'), transform=transform)

# 5. data loaders for batch processing 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # 32 images at once in forward and backward propagation
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 6. pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# 7. Modify the final layer to output 2 classes (real, fake) (Transfer Learning)
num_features = model.fc.in_features 
model.fc = nn.Linear(num_features, 2)

# 8. Move the model to GPU if available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device) 

# 9. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()             
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 10. Set path to save best model in Google Drive
best_model_path = '/content/drive/MyDrive/best_deepfake_detector.pth'

# 11. Training function with validation
def train_model(num_epochs=5):     
    best_val_acc = 0.0             
    for epoch in range(num_epochs):    
        model.train()                   
        running_loss = 0.0
        for images, labels in train_loader:  
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()                
            outputs = model(images)                  # Forward pass
            loss = criterion(outputs, labels)        # Calculate loss
            loss.backward()                          # Backpropagation
            optimizer.step()                         # Update model weights
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset) 

        # Validation phase
        model.eval()   # evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():   # no grad claculations
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device) # images: a tensor of shape [batch_size, 3, 128, 128]
                outputs = model(images) # [-,-]
                _, predicted = torch.max(outputs.data, 1) # Max values (we don't need them, so we use _) ,Indices of the max values, i.e., the predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total         
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

        # Save the best model to Google Drive
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved to Google Drive.")

# 12. Evaluation function on test set
def evaluate_model():
    # Load the best model before testing
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# 13. Run the training and evaluation
train_model(num_epochs=5)
evaluate_model()
