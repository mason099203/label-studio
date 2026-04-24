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

    @staticmethod
    def load_torchscript(
        pt_path: Path,
        device: Optional[str] = None,
    ) -> torch.jit.ScriptModule:
        """
        以指定裝置載入 TorchScript 模型（``best.pt``）。

        ``torch.jit.load`` 若不指定 ``map_location``，在部分 PyTorch 版本中
        會回退至 CPU，即使原本在 GPU 上儲存。本方法統一處理 ``map_location``，
        確保載入後 ``next(model.parameters()).device`` 反映正確裝置。

        @example
        ```python
        model = CNNTrainer.load_torchscript(Path("best.pt"), device="cuda")
        print(next(model.parameters()).device)  # cuda:0
        ```

        @param {Path} pt_path          - TorchScript .pt 檔案路徑
        @param {Optional[str]} device  - 目標裝置："cuda" | "cpu" | None（None 時自動偵測 CUDA）
        @returns {torch.jit.ScriptModule} 位於目標裝置、已設為 eval 模式的 TorchScript 模型
        """
        if device:
            target_device = torch.device(device)
        elif torch.cuda.is_available():
            target_device = torch.device("cuda")
        else:
            target_device = torch.device("cpu")

        # map_location 確保 GPU-saved .pt 在 CPU-only 環境也能載入，
        # 反之亦然：CPU-saved .pt 可透過此參數搬移到 GPU。
        model = torch.jit.load(str(pt_path), map_location=target_device)
        model = model.to(target_device)
        model.eval()

        # 驗證實際裝置（僅在有 parameter/buffer 時有效）
        try:
            actual = str(next(model.parameters()).device)
        except StopIteration:
            actual = str(target_device)

        logger.info(
            "Loaded TorchScript from %s → device=%s (actual=%s)",
            pt_path, target_device, actual,
        )
        return model

    @staticmethod
    def load_checkpoint(
        weights_path: Path,
        num_classes: int,
        device: Optional[str] = None,
    ) -> "CNNClassifier":
        """
        從 ``best.pth`` checkpoint 還原 CNN 模型並搬移至目標裝置。

        使用 ``map_location`` 確保在無 GPU 的環境也能正常載入（GPU checkpoint → CPU）。

        @param {Path} weights_path  - checkpoint 路徑（``best.pth``）
        @param {int} num_classes    - 分類數量（須與訓練時一致）
        @param {Optional[str]} device - 目標裝置："cuda" | "cpu" | None（None 時自動偵測 CUDA）
        @returns {CNNClassifier} 已還原權重且位於目標裝置的模型
        """
        if device:
            target_device = torch.device(device)
        elif torch.cuda.is_available():
            target_device = torch.device("cuda")
        else:
            target_device = torch.device("cpu")

        # map_location 確保 GPU checkpoint 在 CPU-only 環境也能載入
        ckpt = torch.load(weights_path, map_location=target_device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt

        model = CNNClassifier(num_classes=num_classes, pretrained=False)
        model.load_state_dict(state_dict)
        model.to(target_device)
        model.eval()

        logger.info(
            "Loaded checkpoint from %s → device=%s (val_acc=%.2f%%)",
            weights_path,
            target_device,
            ckpt.get("val_acc", float("nan")) if isinstance(ckpt, dict) else float("nan"),
        )
        return model

    def run(self, train_items: List[Dict], val_items: List[Dict]):
        """
        執行完整訓練流程（資料載入 → 訓練 → 驗證 → 匯出最佳模型）。

        GPU 優化項目：
        - ``pin_memory=True``：讓 DataLoader 在 page-locked 記憶體分配資料，加速 CPU→GPU 傳輸。
        - ``non_blocking=True``：讓資料傳輸與 GPU 運算可非同步進行。
        - AMP（混合精度）：訓練時以 FP16 計算降低 GPU 記憶體用量並提升吞吐量。
        - 儲存 ``best.pth`` 時明確呼叫 ``model.to(self.device)``，確保 GPU tensor 被持久化；
          載入時請使用 ``load_checkpoint()`` 並指定 ``map_location`` 以確保跨裝置相容性。
        - TorchScript 於訓練裝置（CUDA / CPU）上 trace，確保 GPU operator path 被正確記錄，
          供 Triton libtorch backend 使用；Triton 端的 instance_group 由 config.pbtxt 決定。

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

                # ── 確保模型位於訓練裝置（CUDA / CPU）後再持久化 ──────────────
                # 明確呼叫 model.to(self.device) 保證 state_dict 中的 tensor
                # 與訓練裝置一致（GPU 時為 CUDA tensor），以利 Triton GPU inference。
                # 載入時請透過 load_checkpoint() 的 map_location 處理跨裝置相容。
                model.to(self.device)
                weights_path = self.run_dir / "best.pth"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "device": str(self.device),
                        "class_names": self.class_names,
                        "class_to_idx": self.class_to_idx,
                        "num_classes": len(self.class_names),
                        "val_acc": v_acc,
                        "epoch": epoch + 1,
                    },
                    weights_path,
                )
                logger.info(
                    "Saved best checkpoint (device=%s, val_acc=%.2f%%) → %s",
                    self.device, v_acc, weights_path,
                )

                # ── TorchScript export ────────────────────────────────────────
                # best.pt     ：在訓練裝置（CUDA / CPU）上 trace，
                #               GPU trace 確保 CUDA operator path 被記錄；
                #               Triton libtorch backend 依 config.pbtxt instance_group
                #               自行搬移 tensor。
                # best_cpu.pt ：強制搬至 CPU 後再 trace，確保 torch.jit.load 後
                #               next(model.parameters()).device == cpu，
                #               供無 GPU 環境或純 Python inference 使用。
                try:
                    model.eval()
                    model.to(self.device)
                    example_dev = torch.randn(1, 3, 224, 224).to(self.device)
                    traced = torch.jit.trace(model, example_dev)
                    traced.save(self.run_dir / "best.pt")
                    logger.info(
                        "Exported TorchScript (device=%s) → %s",
                        self.device, self.run_dir / "best.pt",
                    )

                    # CPU 版：確保 torch.jit.load("best_cpu.pt") 直接在 CPU 上
                    if self.use_cuda:
                        model_cpu = model.to("cpu")
                        example_cpu = example_dev.to("cpu")
                        traced_cpu = torch.jit.trace(model_cpu, example_cpu)
                        traced_cpu.save(self.run_dir / "best_cpu.pt")
                        model.to(self.device)  # 還原回 GPU 繼續訓練
                        logger.info(
                            "Exported TorchScript (device=cpu) → %s",
                            self.run_dir / "best_cpu.pt",
                        )
                except Exception as e:
                    logger.warning("Failed to export TorchScript: %s", e)

        return model, best_acc
