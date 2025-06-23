# Imports
import os  # For directory and path management
from collections import defaultdict  # To group frames by video ID with automatic list creation
from PIL import Image  # To load image files
from tqdm import tqdm  # For progress bars during training/evaluation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights  # Pretrained ResNet18 model


# Parse frame filename into video ID and frame number
def parse_filename(file_name):
    base_name = file_name.rsplit('.', 1)[0]  # Remove extension
    parts = base_name.split('_')  # Split by underscore

    if len(parts) == 4:
        video_id = '_'.join(parts[:3])
        try:
            frame_number = int(parts[3])
            return video_id, frame_number
        except:
            return None, None

    elif len(parts) == 3:
        video_id = '_'.join(parts[:2])
        try:
            frame_number = int(parts[2])
            return video_id, frame_number
        except:
            return None, None

    elif len(parts) == 2:
        video_id = parts[0]
        try:
            frame_number = int(parts[1])
            return video_id, frame_number
        except:
            return None, None

    else:
        return None, None


# Dataset class for loading video frame sequences
class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_frames=50):
        self.videos = []  # List of list of frame paths
        self.labels = []  # List of labels for each video (0=real, 1=fake)
        self.transform = transform  # Image transform
        self.max_frames = max_frames  # Maximum number of frames per video

        for label_name, label in [('fake', 1), ('real', 0)]:
            folder_path = os.path.join(root_dir, label_name)
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} does not exist. Skipping.")
                continue

            video_dict = defaultdict(list)  # Group frames by video ID
            for file in os.listdir(folder_path):
                if not (file.lower().endswith('.jpg') or file.lower().endswith('.png')):
                    continue

                video_id, frame_id = parse_filename(file)
                if video_id is None:
                    print(f"Skipping unknown format file: {file}")
                    continue

                video_dict[video_id].append((frame_id, os.path.join(folder_path, file)))

            for vid, frames in video_dict.items():
                if len(frames) == 0:
                    continue
                sorted_frames = sorted(frames, key=lambda x: x[0])  # Sort by frame number

                # Pad or truncate to max_frames
                if len(sorted_frames) < self.max_frames:
                    sorted_frames += [sorted_frames[-1]] * (self.max_frames - len(sorted_frames))
                else:
                    sorted_frames = sorted_frames[:self.max_frames]

                frame_paths = [f[1] for f in sorted_frames]
                self.videos.append(frame_paths)
                self.labels.append(label)

        print(f"Loaded {len(self.videos)} videos from {root_dir}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        frame_paths = self.videos[idx]
        label = self.labels[idx]
        frames = []
        for path in frame_paths:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        video_tensor = torch.stack(frames)  # Shape: [T, C, H, W]
        return video_tensor, label


# CNN encoder using pre-trained ResNet18
class CNNEncoderResNet(nn.Module):
    def __init__(self, feature_dim=512):
        super(CNNEncoderResNet, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # Remove final FC layer
        self.feature_extractor = nn.Sequential(*modules)  # Feature extractor
        self.fc = nn.Linear(resnet.fc.in_features, feature_dim)  # Feature reduction layer

    def forward(self, x):
        x = self.feature_extractor(x)  # Shape: [B, 512, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [B, 512]
        x = self.fc(x)  # Project to desired feature_dim
        return x


# CNN + RNN model for sequence classification
class CNNRNNModel(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, num_classes=2):
        super(CNNRNNModel, self).__init__()
        self.cnn = CNNEncoderResNet(feature_dim)  # CNN encoder
        self.rnn = nn.LSTM(feature_dim, hidden_dim, batch_first=True)  # RNN to model temporal dynamics
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)  # Merge batch and time
        feats = self.cnn(x)  # Extract CNN features
        feats = feats.view(B, T, -1)  # Reshape to [B, T, feature_dim]
        _, (hn, _) = self.rnn(feats)  # Get last hidden state
        out = self.classifier(hn[-1])  # Pass through classifier
        return out


# Training for one epoch
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for videos, labels in tqdm(loader, desc='Train'):
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100 * correct / total


# Evaluation function (used for validation and testing)
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc='Val/Test'):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100 * correct / total


# Main training loop
def main():
    dataset_path = "/content/destination_folder/1000_videos"  # Change as needed
    max_frames = 50
    batch_size = 4
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                             [0.229, 0.224, 0.225])  # ImageNet std
    ])

    # Load datasets
    train_set = VideoFrameDataset(os.path.join(dataset_path, 'train'), transform, max_frames)
    val_set = VideoFrameDataset(os.path.join(dataset_path, 'validation'), transform, max_frames)
    test_set = VideoFrameDataset(os.path.join(dataset_path, 'test'), transform, max_frames)

    pin_memory = True if torch.cuda.is_available() else False

    # Data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)

    # Model setup
    model = CNNRNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_resnet_lstm_model.pth")
            print(" Model saved.")

    # Final test
    model.load_state_dict(torch.load("best_resnet_lstm_model.pth"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")


# Entry point
if __name__ == "__main__":
    main()
