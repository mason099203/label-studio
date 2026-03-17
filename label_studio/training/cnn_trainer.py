import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Cannot load image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        return image, label

class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super(CNNClassifier, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.dropout(x)
        return x

class CNNTrainer:
    def __init__(
        self, 
        project_id: int,
        run_dir: Path,
        class_names: List[str],
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 50,
        device: Optional[str] = None,
        job_reporter: Any = None
    ):
        self.project_id = project_id
        self.run_dir = run_dir
        self.class_names = class_names
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.job_reporter = job_reporter
        
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f"Using device: {self.device}")

    def create_data_transforms(self):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return train_transform, val_transform

    def train_epoch(self, model, loader, optimizer, criterion):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return running_loss / len(loader), 100 * correct / total

    def validate_epoch(self, model, loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return running_loss / len(loader), 100 * correct / total

    def run(self, train_items: List[Dict], val_items: List[Dict]):
        # data setup
        train_transform, val_transform = self.create_data_transforms()
        
        train_set = ImageClassificationDataset(
            [it['source_image'] for it in train_items],
            [self.class_to_idx[it['label']] for it in train_items],
            train_transform
        )
        val_set = ImageClassificationDataset(
            [it['source_image'] for it in val_items],
            [self.class_to_idx[it['label']] for it in val_items],
            val_transform
        )
        
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        model = CNNClassifier(num_classes=len(self.class_names)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Starting CNN training for {self.num_epochs} epochs")
        best_acc = 0.0
        
        for epoch in range(self.num_epochs):
            t_loss, t_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            v_loss, v_acc = self.validate_epoch(model, val_loader, criterion)
            
            logger.info(f"Epoch {epoch+1}: T-Loss {t_loss:.4f} T-Acc {t_acc:.2f}% | V-Loss {v_loss:.4f} V-Acc {v_acc:.2f}%")
            
            if self.job_reporter:
                self.job_reporter(epoch + 1, self.num_epochs, t_loss, t_acc, v_loss, v_acc)
            
            if v_acc > best_acc:
                best_acc = v_acc
                # Save best weights
                weights_path = self.run_dir / "best.pth"
                torch.save(model.state_dict(), weights_path)
                
                # Also save TorchScript for Triton deployment convenience
                try:
                    model.eval()
                    example = torch.rand(1, 3, 224, 224).to(self.device)
                    traced = torch.jit.trace(model, example)
                    traced.save(self.run_dir / "best.pt")
                except Exception as e:
                    logger.warning(f"Failed to export TorchScript: {e}")

        return model, best_acc
