import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
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
    """
    CNN 分類模型訓練器。
    支援 GPU 自動偵測、混合精度訓練（AMP）與 DataLoader pin_memory 加速。

    @param {int} project_id        - Label Studio 專案 ID
    @param {Path} run_dir          - 訓練結果輸出目錄
    @param {List[str]} class_names - 分類類別名稱清單
    @param {int} batch_size        - 每批次樣本數（預設 32）
    @param {float} learning_rate   - 學習率（預設 0.001）
    @param {int} num_epochs        - 訓練回合數（預設 50）
    @param {Optional[str]} device  - 裝置指定："cuda" | "cpu" | None（None 時自動偵測 CUDA）
    @param {bool} use_amp          - 是否啟用混合精度訓練（僅在 CUDA 裝置上有效，預設 True）
    @param {Any} job_reporter      - 進度回呼函式（可選）
    """

    def __init__(
        self, 
        project_id: int,
        run_dir: Path,
        class_names: List[str],
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 50,
        device: Optional[str] = None,
        use_amp: bool = True,
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

        # 優先使用 CUDA；若明確指定 device 則直接採用
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.use_cuda = self.device.type == "cuda"

        # AMP（混合精度）僅在 CUDA 裝置上才有效
        self.use_amp = use_amp and self.use_cuda
        self.scaler = GradScaler() if self.use_amp else None

        logger.info(
            "Using device: %s | AMP: %s | CUDA device name: %s",
            self.device,
            self.use_amp,
            torch.cuda.get_device_name(self.device) if self.use_cuda else "N/A",
        )

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
        """
        執行單一訓練 epoch，支援 AMP 混合精度以加速 GPU 運算。

        @param model     - PyTorch 模型
        @param loader    - 訓練資料集 DataLoader
        @param optimizer - 優化器
        @param criterion - 損失函式
        @returns {Tuple[float, float]} (平均 loss, 準確率 %)
        """
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(self.device, non_blocking=self.use_cuda)
            labels = labels.to(self.device, non_blocking=self.use_cuda)
            optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
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
        """
        執行單一驗證 epoch。

        @param model     - PyTorch 模型
        @param loader    - 驗證資料集 DataLoader
        @param criterion - 損失函式
        @returns {Tuple[float, float]} (平均 loss, 準確率 %)
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=self.use_cuda)
                labels = labels.to(self.device, non_blocking=self.use_cuda)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return running_loss / len(loader), 100 * correct / total

    def run(self, train_items: List[Dict], val_items: List[Dict]):
        """
        執行完整訓練流程（資料載入 → 訓練 → 驗證 → 匯出最佳模型）。

        GPU 優化項目：
        - ``pin_memory=True``：讓 DataLoader 在 page-locked 記憶體分配資料，加速 CPU→GPU 傳輸。
        - ``non_blocking=True``：讓資料傳輸與 GPU 運算可非同步進行。
        - AMP（混合精度）：訓練時以 FP16 計算降低 GPU 記憶體用量並提升吞吐量。
        - TorchScript 匯出時先搬回 CPU，確保 Triton 可以 CPU/GPU 兩種 instance_group 皆使用。

        @param {List[Dict]} train_items - 訓練集樣本清單
        @param {List[Dict]} val_items   - 驗證集樣本清單
        @returns {Tuple[nn.Module, float]} (最終模型, 最佳驗證準確率)
        """
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

        # pin_memory=True 在 GPU 模式下加速 CPU→GPU 資料傳輸
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.use_cuda,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.use_cuda,
        )
        
        model = CNNClassifier(num_classes=len(self.class_names)).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        logger.info("Starting CNN training for %d epochs on %s", self.num_epochs, self.device)
        best_acc = 0.0
        
        for epoch in range(self.num_epochs):
            t_loss, t_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            v_loss, v_acc = self.validate_epoch(model, val_loader, criterion)
            
            logger.info(
                "Epoch %d/%d: T-Loss %.4f T-Acc %.2f%% | V-Loss %.4f V-Acc %.2f%%",
                epoch + 1, self.num_epochs, t_loss, t_acc, v_loss, v_acc,
            )
            
            if self.job_reporter:
                self.job_reporter(epoch + 1, self.num_epochs, t_loss, t_acc, v_loss, v_acc)
            
            if v_acc > best_acc:
                best_acc = v_acc
                weights_path = self.run_dir / "best.pth"
                torch.save(model.state_dict(), weights_path)
                
                # TorchScript export：在訓練裝置（CUDA / CPU）上 trace，
                # 以確保 GPU operator path 被正確記錄進 TorchScript。
                # Triton 載入後的 instance_group 由 config.pbtxt 決定，
                # libtorch backend 會自行搬移 tensor 至對應裝置。
                try:
                    model.eval()
                    model.to(self.device)
                    example = torch.randn(1, 3, 224, 224).to(self.device)
                    traced = torch.jit.trace(model, example)
                    traced.save(self.run_dir / "best.pt")
                except Exception as e:
                    logger.warning("Failed to export TorchScript: %s", e)

        return model, best_acc
