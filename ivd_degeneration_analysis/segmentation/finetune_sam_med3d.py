"""
SAM-Med3D 椎间盘分割微调脚本

基于 SAM-Med3D 官方训练代码改写，专门用于椎间盘分割任务

使用方法:
    # 1. 先准备数据
    python prepare_finetune_data.py /path/to/labeled_data /path/to/sam_data

    # 2. 运行微调
    python finetune_sam_med3d.py \
        --data-dir /path/to/sam_data/ivd/mri_IVD \
        --checkpoint /path/to/sam_med3d_turbo.pth \
        --output-dir ./ivd_finetune_output \
        --epochs 50 \
        --batch-size 2 \
        --lr 1e-4

    # 3. 使用微调后的模型进行分割
    python interactive_segment.py image.nii.gz \
        --checkpoint ./ivd_finetune_output/best_model.pth
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import nibabel as nib
import torchio as tio
from monai.losses import DiceCELoss

# 添加 SAM-Med3D 路径
SAM_MED3D_PATH = "/home/nyuair/junyi/SAM-Med3D"
if SAM_MED3D_PATH not in sys.path:
    sys.path.insert(0, SAM_MED3D_PATH)


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IVDDataset(Dataset):
    """椎间盘分割数据集"""

    def __init__(
        self,
        data_dir: str,
        img_size: int = 128,
        transform: Optional[tio.Transform] = None,
        num_clicks: int = 5,
        is_train: bool = True
    ):
        """
        Args:
            data_dir: 数据目录（包含 imagesTr 和 labelsTr 子目录）
            img_size: 图像尺寸
            transform: 数据增强变换
            num_clicks: 每个样本的点击数
            is_train: 是否为训练集
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        self.num_clicks = num_clicks
        self.is_train = is_train

        # 加载数据路径
        images_dir = os.path.join(data_dir, "imagesTr")
        labels_dir = os.path.join(data_dir, "labelsTr")

        self.image_paths = sorted([
            os.path.join(images_dir, f) for f in os.listdir(images_dir)
            if f.endswith('.nii.gz') or f.endswith('.nii')
        ])

        self.label_paths = []
        for img_path in self.image_paths:
            img_name = os.path.basename(img_path)
            label_path = os.path.join(labels_dir, img_name)
            if os.path.exists(label_path):
                self.label_paths.append(label_path)
            else:
                logger.warning(f"找不到标签文件: {label_path}")
                self.image_paths.remove(img_path)

        logger.info(f"加载了 {len(self.image_paths)} 个样本")

        # 默认数据增强
        if transform is None and is_train:
            self.transform = tio.Compose([
                tio.RandomFlip(axes=(0, 1, 2), p=0.5),
                tio.RandomAffine(
                    scales=(0.9, 1.1),
                    degrees=10,
                    p=0.5
                ),
                tio.RandomNoise(std=(0, 0.1), p=0.3),
                tio.RandomBlur(std=(0, 1), p=0.3),
            ])

        # 归一化
        self.normalize = tio.ZNormalization(masking_method=lambda x: x > 0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载数据
        image = nib.load(self.image_paths[idx]).get_fdata().astype(np.float32)
        label = nib.load(self.label_paths[idx]).get_fdata().astype(np.int64)

        # 确保维度正确
        if image.ndim == 3:
            image = image[np.newaxis, ...]  # [C, D, H, W]
        if label.ndim == 3:
            label = label[np.newaxis, ...]

        # 创建 TorchIO Subject
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.from_numpy(image)),
            label=tio.LabelMap(tensor=torch.from_numpy(label))
        )

        # 应用数据增强
        if self.transform:
            subject = self.transform(subject)

        # 归一化
        subject = self.normalize(subject)

        # 调整尺寸
        resize = tio.CropOrPad(
            target_shape=(self.img_size, self.img_size, self.img_size),
            mask_name='label'
        )
        subject = resize(subject)

        image_tensor = subject['image'].data.float()  # [1, D, H, W]
        label_tensor = subject['label'].data.long()   # [1, D, H, W]

        # 生成随机点击点
        click_points, click_labels = self._generate_click_points(label_tensor[0])

        return {
            'image': image_tensor,
            'label': label_tensor,
            'click_points': click_points,
            'click_labels': click_labels,
            'name': os.path.basename(self.image_paths[idx])
        }

    def _generate_click_points(
        self,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成随机点击点

        Args:
            label: 标签张量 [D, H, W]

        Returns:
            (points, labels) - points: [N, 3], labels: [N]
        """
        # 前景点
        fg_coords = torch.where(label > 0)
        if len(fg_coords[0]) > 0:
            num_fg = min(self.num_clicks, len(fg_coords[0]))
            indices = torch.randperm(len(fg_coords[0]))[:num_fg]
            fg_points = torch.stack([
                fg_coords[0][indices],
                fg_coords[1][indices],
                fg_coords[2][indices]
            ], dim=1).float()
            fg_labels = torch.ones(num_fg, dtype=torch.long)
        else:
            fg_points = torch.zeros(0, 3)
            fg_labels = torch.zeros(0, dtype=torch.long)

        # 背景点（可选）
        bg_coords = torch.where(label == 0)
        if len(bg_coords[0]) > 0 and self.is_train:
            num_bg = max(1, self.num_clicks // 2)
            indices = torch.randperm(len(bg_coords[0]))[:num_bg]
            bg_points = torch.stack([
                bg_coords[0][indices],
                bg_coords[1][indices],
                bg_coords[2][indices]
            ], dim=1).float()
            bg_labels = torch.zeros(num_bg, dtype=torch.long)
        else:
            bg_points = torch.zeros(0, 3)
            bg_labels = torch.zeros(0, dtype=torch.long)

        # 合并
        points = torch.cat([fg_points, bg_points], dim=0)
        labels = torch.cat([fg_labels, bg_labels], dim=0)

        # 填充到固定长度
        max_points = self.num_clicks * 2
        if len(points) < max_points:
            padding = torch.zeros(max_points - len(points), 3)
            points = torch.cat([points, padding], dim=0)
            padding_labels = torch.zeros(max_points - len(labels), dtype=torch.long)
            labels = torch.cat([labels, padding_labels], dim=0)

        return points[:max_points], labels[:max_points]


class SAMMed3DFineTuner:
    """SAM-Med3D 微调器"""

    def __init__(
        self,
        checkpoint_path: str,
        output_dir: str,
        device: str = "cuda",
        freeze_encoder: bool = False
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.freeze_encoder = freeze_encoder
        os.makedirs(output_dir, exist_ok=True)

        # 加载模型
        self._load_model(checkpoint_path)

        # 损失函数
        self.loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    def _load_model(self, checkpoint_path: str):
        """加载预训练模型"""
        try:
            from segment_anything.build_sam3D import sam_model_registry3D

            logger.info(f"加载模型: {checkpoint_path}")

            # 创建模型
            self.model = sam_model_registry3D['vit_b_ori'](checkpoint=None)

            # 加载权重
            if checkpoint_path and os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("预训练权重加载成功")
            else:
                logger.warning("未找到预训练权重，从头开始训练")

            self.model = self.model.to(self.device)

            # 冻结编码器（可选）
            if self.freeze_encoder:
                for param in self.model.image_encoder.parameters():
                    param.requires_grad = False
                logger.info("图像编码器已冻结")

        except ImportError as e:
            logger.error(f"无法导入 SAM-Med3D: {e}")
            logger.error(f"请确保 {SAM_MED3D_PATH} 路径正确")
            raise

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        accumulation_steps: int = 4,
        save_every: int = 10
    ):
        """训练模型"""
        # 优化器
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        # 学习率调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler()

        best_dice = 0.0
        train_losses = []
        val_dices = []

        for epoch in range(epochs):
            # 训练
            self.model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for step, batch in enumerate(pbar):
                image = batch['image'].to(self.device)
                label = batch['label'].to(self.device)
                click_points = batch['click_points'].to(self.device)
                click_labels = batch['click_labels'].to(self.device)

                with torch.cuda.amp.autocast():
                    # 前向传播
                    image_embedding = self.model.image_encoder(image)

                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=(click_points, click_labels),
                        boxes=None,
                        masks=None
                    )

                    low_res_masks, iou_predictions = self.model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False
                    )

                    # 上采样到原始尺寸
                    masks = F.interpolate(
                        low_res_masks,
                        size=label.shape[2:],
                        mode='trilinear',
                        align_corners=False
                    )

                    # 计算损失
                    loss = self.loss_fn(masks, label.float())
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * accumulation_steps
                pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)

            # 验证
            if val_loader is not None:
                val_dice = self.validate(val_loader)
                val_dices.append(val_dice)
                logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_dice={val_dice:.4f}")

                # 保存最佳模型
                if val_dice > best_dice:
                    best_dice = val_dice
                    self._save_checkpoint(
                        os.path.join(self.output_dir, "best_model.pth"),
                        epoch, optimizer
                    )
            else:
                logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

            # 定期保存
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(
                    os.path.join(self.output_dir, f"checkpoint_epoch{epoch+1}.pth"),
                    epoch, optimizer
                )

            scheduler.step()

        # 保存最终模型
        self._save_checkpoint(
            os.path.join(self.output_dir, "final_model.pth"),
            epochs - 1, optimizer
        )

        # 保存训练曲线
        self._save_training_curves(train_losses, val_dices)

        return train_losses, val_dices

    def validate(self, val_loader: DataLoader) -> float:
        """验证模型"""
        self.model.eval()
        dice_scores = []

        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(self.device)
                label = batch['label'].to(self.device)
                click_points = batch['click_points'].to(self.device)
                click_labels = batch['click_labels'].to(self.device)

                # 前向传播
                image_embedding = self.model.image_encoder(image)

                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=(click_points, click_labels),
                    boxes=None,
                    masks=None
                )

                low_res_masks, _ = self.model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )

                # 上采样
                masks = F.interpolate(
                    low_res_masks,
                    size=label.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )

                # 计算 Dice
                pred = (torch.sigmoid(masks) > 0.5).float()
                dice = self._compute_dice(pred, label.float())
                dice_scores.append(dice.item())

        return np.mean(dice_scores)

    def _compute_dice(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 Dice 系数"""
        smooth = 1e-5
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    def _save_checkpoint(self, path: str, epoch: int, optimizer):
        """保存检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        logger.info(f"保存检查点: {path}")

    def _save_training_curves(self, train_losses: List[float], val_dices: List[float]):
        """保存训练曲线"""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(train_losses)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')

        if val_dices:
            ax2.plot(val_dices)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Dice')
            ax2.set_title('Validation Dice')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="SAM-Med3D 椎间盘分割微调")
    parser.add_argument("--data-dir", required=True, help="数据目录")
    parser.add_argument("--checkpoint", default=None, help="预训练模型路径")
    parser.add_argument("--output-dir", default="./ivd_finetune_output", help="输出目录")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=2, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--img-size", type=int, default=128, help="图像尺寸")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--accumulation-steps", type=int, default=4, help="梯度累积步数")

    # 模型参数
    parser.add_argument("--freeze-encoder", action="store_true", help="冻结图像编码器")
    parser.add_argument("--num-clicks", type=int, default=5, help="点击点数量")

    # 验证参数
    parser.add_argument("--val-split", type=float, default=0.2, help="验证集比例")

    args = parser.parse_args()

    # 创建数据集
    full_dataset = IVDDataset(
        args.data_dir,
        img_size=args.img_size,
        num_clicks=args.num_clicks,
        is_train=True
    )

    # 划分训练集和验证集
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    ) if val_size > 0 else None

    logger.info(f"训练集: {train_size} 样本, 验证集: {val_size} 样本")

    # 创建微调器
    finetuner = SAMMed3DFineTuner(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        freeze_encoder=args.freeze_encoder
    )

    # 开始训练
    finetuner.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        accumulation_steps=args.accumulation_steps
    )


if __name__ == "__main__":
    main()
