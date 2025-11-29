"""
SAM-Med3D 交互式椎间盘分割工具

使用方法:
1. 准备 NIfTI 格式的 MRI 图像
2. 运行此脚本，在 3D 视图中点击椎间盘区域作为 prompt
3. 模型自动生成分割结果
4. 可选：手动修正并保存

依赖:
pip install medim torch torchio nibabel matplotlib
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from dataclasses import dataclass

# 导入配置
try:
    from config import SAM_MED3D_PATH, SAM_MED3D_CHECKPOINT, MASK_OUTPUT_DIR, DISC_LABELS
except ImportError:
    SAM_MED3D_PATH = "/home/nyuair/junyi/SAM-Med3D"
    SAM_MED3D_CHECKPOINT = None
    MASK_OUTPUT_DIR = "./output/masks"
    DISC_LABELS = {
        'L1-L2': {'disc': 3}, 'L2-L3': {'disc': 5}, 'L3-L4': {'disc': 7},
        'L4-L5': {'disc': 9}, 'L5-S1': {'disc': 11}
    }

# 添加 SAM-Med3D 路径
if SAM_MED3D_PATH not in sys.path:
    sys.path.insert(0, SAM_MED3D_PATH)


@dataclass
class PromptPoint:
    """提示点数据结构"""
    x: int
    y: int
    z: int
    label: int  # 1=前景, 0=背景


class SAMMed3DSegmenter:
    """SAM-Med3D 分割器封装"""

    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.checkpoint_path = checkpoint_path

        self._load_model()

    def _load_model(self):
        """加载 SAM-Med3D 模型"""
        try:
            import medim

            if self.checkpoint_path is None:
                # 使用 MedIM 自动下载预训练权重
                ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
                print(f"[INFO] 正在从 HuggingFace 加载 SAM-Med3D-turbo 模型...")
                self.model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
            else:
                print(f"[INFO] 从本地加载模型: {self.checkpoint_path}")
                self.model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=self.checkpoint_path)

            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"[SUCCESS] 模型加载成功，使用设备: {self.device}")

        except ImportError:
            print("[ERROR] 请先安装 medim: pip install medim")
            raise

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理图像

        Args:
            image: 3D numpy array [D, H, W]

        Returns:
            预处理后的 tensor
        """
        # 归一化到 0-1
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # 转换为 tensor [B, C, D, H, W]
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

        return image_tensor.to(self.device)

    def segment_with_prompts(
        self,
        image: np.ndarray,
        prompt_points: List[PromptPoint],
        num_clicks: int = 1
    ) -> np.ndarray:
        """
        使用提示点进行分割

        Args:
            image: 3D numpy array [D, H, W]
            prompt_points: 提示点列表
            num_clicks: 每次使用的点击数

        Returns:
            分割掩码 [D, H, W]
        """
        if not prompt_points:
            raise ValueError("需要至少一个提示点")

        # 预处理图像
        image_tensor = self.preprocess_image(image)

        # 准备提示点
        coords = []
        labels = []
        for pt in prompt_points:
            coords.append([pt.x, pt.y, pt.z])
            labels.append(pt.label)

        coords = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(self.device)
        labels = torch.tensor(labels, dtype=torch.int64).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 编码图像
            image_embedding = self.model.image_encoder(image_tensor)

            # 编码提示
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=(coords, labels),
                boxes=None,
                masks=None
            )

            # 解码生成掩码
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )

            # 后处理
            masks = torch.nn.functional.interpolate(
                low_res_masks,
                size=image.shape,
                mode='trilinear',
                align_corners=False
            )
            masks = masks.squeeze().cpu().numpy()
            masks = (masks > 0.5).astype(np.uint8)

        return masks


class InteractiveSegmentationGUI:
    """交互式分割 GUI"""

    def __init__(
        self,
        image_path: str,
        segmenter: SAMMed3DSegmenter,
        output_dir: str = "./output"
    ):
        self.image_path = image_path
        self.segmenter = segmenter
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 加载图像
        self.nii = nib.load(image_path)
        self.image = self.nii.get_fdata()
        self.affine = self.nii.affine

        # 初始化状态
        self.prompt_points: List[PromptPoint] = []
        self.current_mask: Optional[np.ndarray] = None
        self.current_slice = [
            self.image.shape[0] // 2,
            self.image.shape[1] // 2,
            self.image.shape[2] // 2
        ]

        # 椎间盘标签配置
        self.disc_labels = {
            'L1-L2': 3,
            'L2-L3': 5,
            'L3-L4': 7,
            'L4-L5': 9,
            'L5-S1': 11
        }
        self.current_disc = 'L4-L5'  # 默认分割 L4-L5

        self._setup_gui()

    def _setup_gui(self):
        """设置 GUI"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('SAM-Med3D 椎间盘交互式分割', fontsize=14)

        # 创建三视图
        self.ax_sag = self.fig.add_subplot(2, 3, 1)  # 矢状位
        self.ax_cor = self.fig.add_subplot(2, 3, 2)  # 冠状位
        self.ax_axi = self.fig.add_subplot(2, 3, 3)  # 轴位

        # 创建控制面板区域
        self.ax_info = self.fig.add_subplot(2, 3, 4)
        self.ax_info.axis('off')

        # 滑块区域
        ax_slider_sag = plt.axes([0.1, 0.15, 0.2, 0.03])
        ax_slider_cor = plt.axes([0.4, 0.15, 0.2, 0.03])
        ax_slider_axi = plt.axes([0.7, 0.15, 0.2, 0.03])

        self.slider_sag = Slider(ax_slider_sag, '矢状位', 0, self.image.shape[0]-1,
                                  valinit=self.current_slice[0], valstep=1)
        self.slider_cor = Slider(ax_slider_cor, '冠状位', 0, self.image.shape[1]-1,
                                  valinit=self.current_slice[1], valstep=1)
        self.slider_axi = Slider(ax_slider_axi, '轴位', 0, self.image.shape[2]-1,
                                  valinit=self.current_slice[2], valstep=1)

        # 按钮
        ax_btn_segment = plt.axes([0.1, 0.05, 0.15, 0.05])
        ax_btn_clear = plt.axes([0.3, 0.05, 0.15, 0.05])
        ax_btn_save = plt.axes([0.5, 0.05, 0.15, 0.05])
        ax_btn_next = plt.axes([0.7, 0.05, 0.15, 0.05])

        self.btn_segment = Button(ax_btn_segment, '分割')
        self.btn_clear = Button(ax_btn_clear, '清除点')
        self.btn_save = Button(ax_btn_save, '保存结果')
        self.btn_next = Button(ax_btn_next, '下一个椎间盘')

        # 绑定事件
        self.slider_sag.on_changed(self._on_slider_change)
        self.slider_cor.on_changed(self._on_slider_change)
        self.slider_axi.on_changed(self._on_slider_change)

        self.btn_segment.on_clicked(self._on_segment)
        self.btn_clear.on_clicked(self._on_clear)
        self.btn_save.on_clicked(self._on_save)
        self.btn_next.on_clicked(self._on_next_disc)

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        self._update_display()

    def _on_slider_change(self, val):
        """滑块变化回调"""
        self.current_slice[0] = int(self.slider_sag.val)
        self.current_slice[1] = int(self.slider_cor.val)
        self.current_slice[2] = int(self.slider_axi.val)
        self._update_display()

    def _on_click(self, event):
        """鼠标点击回调"""
        if event.inaxes == self.ax_sag:
            # 矢状位点击
            x, y = int(event.xdata), int(event.ydata)
            z = self.current_slice[0]
            self._add_prompt_point(z, y, x, event.button)

        elif event.inaxes == self.ax_cor:
            # 冠状位点击
            x, z = int(event.xdata), int(event.ydata)
            y = self.current_slice[1]
            self._add_prompt_point(z, y, x, event.button)

        elif event.inaxes == self.ax_axi:
            # 轴位点击
            x, y = int(event.xdata), int(event.ydata)
            z = self.current_slice[2]
            self._add_prompt_point(z, y, x, event.button)

    def _add_prompt_point(self, z, y, x, button):
        """添加提示点"""
        label = 1 if button == 1 else 0  # 左键前景，右键背景
        point = PromptPoint(x=x, y=y, z=z, label=label)
        self.prompt_points.append(point)
        print(f"[INFO] 添加{'前景' if label == 1 else '背景'}点: ({x}, {y}, {z})")
        self._update_display()

    def _on_segment(self, event):
        """执行分割"""
        if not self.prompt_points:
            print("[WARNING] 请先添加提示点")
            return

        print(f"[INFO] 使用 {len(self.prompt_points)} 个提示点进行分割...")

        try:
            self.current_mask = self.segmenter.segment_with_prompts(
                self.image,
                self.prompt_points
            )
            print("[SUCCESS] 分割完成")
            self._update_display()
        except Exception as e:
            print(f"[ERROR] 分割失败: {e}")

    def _on_clear(self, event):
        """清除提示点"""
        self.prompt_points.clear()
        self.current_mask = None
        print("[INFO] 已清除所有提示点")
        self._update_display()

    def _on_save(self, event):
        """保存分割结果"""
        if self.current_mask is None:
            print("[WARNING] 没有分割结果可保存")
            return

        case_name = Path(self.image_path).stem.replace('.nii', '')
        mask_path = os.path.join(
            self.output_dir,
            f"{case_name}_{self.current_disc}_mask.nii.gz"
        )

        # 创建带标签的掩码
        labeled_mask = self.current_mask * self.disc_labels[self.current_disc]

        mask_nii = nib.Nifti1Image(labeled_mask.astype(np.uint8), self.affine)
        nib.save(mask_nii, mask_path)
        print(f"[SUCCESS] 已保存分割结果到: {mask_path}")

    def _on_next_disc(self, event):
        """切换到下一个椎间盘"""
        disc_list = list(self.disc_labels.keys())
        current_idx = disc_list.index(self.current_disc)
        next_idx = (current_idx + 1) % len(disc_list)
        self.current_disc = disc_list[next_idx]

        self.prompt_points.clear()
        self.current_mask = None
        print(f"[INFO] 切换到椎间盘: {self.current_disc}")
        self._update_display()

    def _update_display(self):
        """更新显示"""
        # 清除旧图像
        self.ax_sag.clear()
        self.ax_cor.clear()
        self.ax_axi.clear()

        z, y, x = self.current_slice

        # 显示三视图
        self.ax_sag.imshow(self.image[z, :, :], cmap='gray', aspect='auto')
        self.ax_sag.set_title(f'矢状位 (Z={z})')

        self.ax_cor.imshow(self.image[:, y, :], cmap='gray', aspect='auto')
        self.ax_cor.set_title(f'冠状位 (Y={y})')

        self.ax_axi.imshow(self.image[:, :, x], cmap='gray', aspect='auto')
        self.ax_axi.set_title(f'轴位 (X={x})')

        # 叠加分割结果
        if self.current_mask is not None:
            alpha = 0.4
            self.ax_sag.imshow(self.current_mask[z, :, :], cmap='Reds', alpha=alpha, aspect='auto')
            self.ax_cor.imshow(self.current_mask[:, y, :], cmap='Reds', alpha=alpha, aspect='auto')
            self.ax_axi.imshow(self.current_mask[:, :, x], cmap='Reds', alpha=alpha, aspect='auto')

        # 显示提示点
        for pt in self.prompt_points:
            color = 'g' if pt.label == 1 else 'r'
            marker = 'o' if pt.label == 1 else 'x'

            if pt.z == z:
                self.ax_sag.plot(pt.x, pt.y, marker, color=color, markersize=10)
            if pt.y == y:
                self.ax_cor.plot(pt.x, pt.z, marker, color=color, markersize=10)
            if pt.x == x:
                self.ax_axi.plot(pt.y, pt.z, marker, color=color, markersize=10)

        # 更新信息面板
        self.ax_info.clear()
        self.ax_info.axis('off')
        info_text = f"""
当前椎间盘: {self.current_disc} (标签: {self.disc_labels[self.current_disc]})
提示点数量: {len(self.prompt_points)}
图像尺寸: {self.image.shape}

操作说明:
- 左键点击: 添加前景点 (绿色)
- 右键点击: 添加背景点 (红色)
- 分割: 使用提示点生成分割
- 清除点: 清除所有提示点
- 保存结果: 保存当前分割掩码
- 下一个椎间盘: 切换目标椎间盘
        """
        self.ax_info.text(0.1, 0.9, info_text, transform=self.ax_info.transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='sans-serif')

        self.fig.canvas.draw_idle()

    def run(self):
        """运行 GUI"""
        plt.show()


def simple_segment(
    image_path: str,
    output_path: str,
    prompt_points: List[Tuple[int, int, int, int]],
    checkpoint_path: str = None
):
    """
    简化版分割接口（非交互式）

    Args:
        image_path: 输入图像路径
        output_path: 输出掩码路径
        prompt_points: 提示点列表 [(x, y, z, label), ...]
        checkpoint_path: 模型权重路径
    """
    # 加载模型
    segmenter = SAMMed3DSegmenter(checkpoint_path=checkpoint_path)

    # 加载图像
    nii = nib.load(image_path)
    image = nii.get_fdata()

    # 转换提示点格式
    points = [PromptPoint(x=p[0], y=p[1], z=p[2], label=p[3]) for p in prompt_points]

    # 分割
    mask = segmenter.segment_with_prompts(image, points)

    # 保存
    mask_nii = nib.Nifti1Image(mask.astype(np.uint8), nii.affine)
    nib.save(mask_nii, output_path)
    print(f"[SUCCESS] 分割结果已保存到: {output_path}")

    return mask


def main():
    parser = argparse.ArgumentParser(description="SAM-Med3D 椎间盘交互式分割")
    parser.add_argument("image", help="输入 NIfTI 图像路径")
    parser.add_argument("--output-dir", "-o", default="./segmentation_output",
                        help="输出目录")
    parser.add_argument("--checkpoint", "-c", default=None,
                        help="SAM-Med3D 模型权重路径")
    parser.add_argument("--non-interactive", action="store_true",
                        help="非交互模式（需要提供提示点）")
    parser.add_argument("--points", nargs="+", type=int,
                        help="非交互模式的提示点 (x1 y1 z1 label1 x2 y2 z2 label2 ...)")

    args = parser.parse_args()

    if args.non_interactive:
        if not args.points or len(args.points) % 4 != 0:
            print("[ERROR] 非交互模式需要提供提示点，格式: x1 y1 z1 label1 x2 y2 z2 label2 ...")
            return

        points = []
        for i in range(0, len(args.points), 4):
            points.append(tuple(args.points[i:i+4]))

        output_path = os.path.join(args.output_dir,
                                   Path(args.image).stem.replace('.nii', '') + "_mask.nii.gz")
        simple_segment(args.image, output_path, points, args.checkpoint)
    else:
        # 交互模式
        segmenter = SAMMed3DSegmenter(checkpoint_path=args.checkpoint)
        gui = InteractiveSegmentationGUI(args.image, segmenter, args.output_dir)
        gui.run()


if __name__ == "__main__":
    main()
