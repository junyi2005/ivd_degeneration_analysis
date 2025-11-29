#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化分割结果
生成PNG切片图展示原图与分割叠加效果
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob


def normalize_image(img):
    """归一化图像到0-1范围"""
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img[img > 0], [1, 99]) if np.any(img > 0) else (0, 1)
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-8)
    return img


def visualize_segmentation(img_path, seg_path, output_dir, num_slices=5):
    """
    可视化分割结果

    Args:
        img_path: 原始图像路径
        seg_path: 分割结果路径
        output_dir: 输出目录
        num_slices: 展示的切片数量
    """
    # 加载数据
    img_nib = nib.load(img_path)
    img_data = img_nib.get_fdata()

    seg_nib = nib.load(seg_path)
    seg_data = seg_nib.get_fdata()

    # 获取文件名
    base_name = os.path.basename(img_path).replace('.nii.gz', '')

    # 归一化图像
    img_norm = normalize_image(img_data)

    # 选择要展示的切片（等间距选择）
    z_dim = img_data.shape[2]
    slice_indices = np.linspace(z_dim // 4, 3 * z_dim // 4, num_slices, dtype=int)

    # 创建图形
    fig, axes = plt.subplots(2, num_slices, figsize=(4 * num_slices, 8))

    for i, z in enumerate(slice_indices):
        # 原图
        axes[0, i].imshow(img_norm[:, :, z].T, cmap='gray', origin='lower')
        axes[0, i].set_title(f'Slice {z}')
        axes[0, i].axis('off')

        # 叠加分割
        axes[1, i].imshow(img_norm[:, :, z].T, cmap='gray', origin='lower')
        seg_slice = seg_data[:, :, z].T
        if np.any(seg_slice > 0):
            # 创建叠加mask
            overlay = np.ma.masked_where(seg_slice == 0, seg_slice)
            axes[1, i].imshow(overlay, cmap='hot', alpha=0.5, origin='lower')
        axes[1, i].set_title(f'With Segmentation')
        axes[1, i].axis('off')

    plt.suptitle(f'{base_name}\nTop: Original | Bottom: With Segmentation Overlay', fontsize=14)
    plt.tight_layout()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{base_name}_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'保存可视化: {output_path}')
    return output_path


def visualize_all_results():
    """可视化所有分割结果"""
    nifti_dir = "/home/nyuair/junyi/SAM-Med3D/9hospital/nifti"
    seg_dir = "/home/nyuair/junyi/SAM-Med3D/9hospital/segmentation_results_improved"
    vis_dir = "/home/nyuair/junyi/SAM-Med3D/9hospital/visualizations"

    for patient_dir in sorted(glob(os.path.join(seg_dir, "*"))):
        patient = os.path.basename(patient_dir)
        print(f"\n处理病人: {patient}")

        for seg_path in glob(os.path.join(patient_dir, "*_seg.nii.gz")):
            seq_name = os.path.basename(seg_path).replace('_seg.nii.gz', '')
            img_path = os.path.join(nifti_dir, patient, f"{seq_name}.nii.gz")

            if os.path.exists(img_path):
                patient_vis_dir = os.path.join(vis_dir, patient)
                visualize_segmentation(img_path, seg_path, patient_vis_dir)
            else:
                print(f"  找不到原图: {img_path}")


def create_summary_figure():
    """创建所有病人的摘要图"""
    vis_dir = "/home/nyuair/junyi/SAM-Med3D/9hospital/visualizations"
    all_vis = sorted(glob(os.path.join(vis_dir, "*/*.png")))

    if not all_vis:
        print("没有找到可视化结果")
        return

    # 创建摘要
    n_images = len(all_vis)
    fig, axes = plt.subplots(1, n_images, figsize=(6 * n_images, 8))
    if n_images == 1:
        axes = [axes]

    for i, vis_path in enumerate(all_vis):
        img = plt.imread(vis_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(os.path.basename(os.path.dirname(vis_path)), fontsize=10)

    plt.suptitle('SAM-Med3D Spine MRI Segmentation Results', fontsize=16)
    plt.tight_layout()

    summary_path = os.path.join(vis_dir, 'summary.png')
    plt.savefig(summary_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n保存摘要图: {summary_path}")


if __name__ == "__main__":
    visualize_all_results()
    create_summary_figure()
