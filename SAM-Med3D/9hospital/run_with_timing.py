#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM-Med3D 脊柱MRI分割 - 带时间统计
"""

import os
import sys
import time
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import torchio as tio
from glob import glob
from datetime import datetime

sys.path.insert(0, '/home/nyuair/junyi/SAM-Med3D')
import medim

# 全局结果收集
results = []

def get_model(ckpt_path=None):
    if ckpt_path is None:
        ckpt_path = "/home/nyuair/junyi/SAM-Med3D/ckpt/sam_med3d_turbo.pth"
    print(f"加载模型: {ckpt_path}")
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
    return model


def preprocess_image(img_path, crop_size=128, target_spacing=(1.5, 1.5, 1.5)):
    sitk_image = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(sitk_image)

    meta_info = {
        "sitk_image_object": sitk_image,
        "sitk_origin": sitk_image.GetOrigin(),
        "sitk_direction": sitk_image.GetDirection(),
        "sitk_spacing": sitk_image.GetSpacing(),
        "original_numpy_shape": img_arr.shape,
    }

    subject = tio.Subject(image=tio.ScalarImage(img_path))
    meta_info["original_subject_affine"] = subject.image.affine.copy()
    meta_info["original_subject_spatial_shape"] = subject.image.spatial_shape

    resampler = tio.Resample(target=target_spacing)
    subject_resampled = resampler(subject)

    transform_canonical = tio.ToCanonical()
    subject_canonical = transform_canonical(subject_resampled)

    meta_info["canonical_subject_shape"] = subject_canonical.spatial_shape
    meta_info["canonical_subject_affine"] = subject_canonical.image.affine.copy()

    crop_transform = tio.CropOrPad(target_shape=(crop_size, crop_size, crop_size))
    padding_params, cropping_params = crop_transform._compute_center_crop_or_pad(subject_canonical)
    subject_cropped = crop_transform(subject_canonical)

    meta_info["padding_params_functional"] = padding_params
    meta_info["cropping_params_functional"] = cropping_params
    meta_info["roi_subject_affine"] = subject_cropped.image.affine.copy()

    img3D_roi = subject_cropped.image.data.clone().detach()
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    img3D_roi = norm_transform(img3D_roi.squeeze(dim=1))
    img3D_roi = img3D_roi.unsqueeze(dim=1)

    if img3D_roi.ndim == 4:
        img3D_roi = img3D_roi.unsqueeze(0)

    return img3D_roi, meta_info


def infer_with_points(model, roi_image, point_coords, point_labels, num_iterations=3):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        input_tensor = roi_image.to(device)
        image_embeddings = model.image_encoder(input_tensor)

        points_coords = torch.tensor(point_coords, dtype=torch.float32, device=device).reshape(1, -1, 3)
        points_labels = torch.tensor(point_labels, dtype=torch.int64, device=device).reshape(1, -1)

        prev_low_res_mask = torch.zeros(1, 1,
                                        roi_image.shape[2] // 4,
                                        roi_image.shape[3] // 4,
                                        roi_image.shape[4] // 4,
                                        device=device, dtype=torch.float)

        for _ in range(num_iterations):
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=[points_coords, points_labels],
                boxes=None,
                masks=prev_low_res_mask,
            )

            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
            prev_low_res_mask = low_res_masks.detach()

        final_masks_hr = F.interpolate(
            low_res_masks,
            size=roi_image.shape[-3:],
            mode='trilinear',
            align_corners=False
        )

    medsam_seg_prob = torch.sigmoid(final_masks_hr)
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg_mask = (medsam_seg_prob > 0.5).astype(np.uint8)

    return medsam_seg_mask


def postprocess_mask(roi_pred_numpy, meta_info):
    roi_pred_tensor = torch.from_numpy(roi_pred_numpy.astype(np.float32)).unsqueeze(0)

    pred_label_map = tio.LabelMap(tensor=roi_pred_tensor, affine=meta_info["roi_subject_affine"])

    reference_tensor_shape = (1, *meta_info["original_subject_spatial_shape"])
    reference_image = tio.ScalarImage(
        tensor=torch.zeros(reference_tensor_shape),
        affine=meta_info["original_subject_affine"]
    )

    resampler = tio.Resample(target=reference_image, image_interpolation='nearest')
    pred_resampled = resampler(pred_label_map)
    final_pred = pred_resampled.data.squeeze(0).cpu().numpy()

    return final_pred.transpose(2, 1, 0).astype(np.uint8)


def save_prediction(pred_arr, output_path, meta_info):
    out_img = sitk.GetImageFromArray(pred_arr)
    sitk_img = meta_info.get("sitk_image_object")
    if sitk_img:
        out_img.SetOrigin(sitk_img.GetOrigin())
        out_img.SetDirection(sitk_img.GetDirection())
        out_img.SetSpacing(sitk_img.GetSpacing())
    sitk.WriteImage(out_img, output_path)


def get_spine_prompt_points(image_shape=(128, 128, 128)):
    h, w, d = image_shape
    points = []
    labels = []

    for z in [d//4, d//2, 3*d//4]:
        points.append([w//2, h//2, z])
        points.append([w//2, h//2-10, z])
        labels.extend([1, 1])

    points.append([10, 10, d//2])
    points.append([w-10, 10, d//2])
    points.append([10, h-10, d//2])
    points.append([w-10, h-10, d//2])
    labels.extend([0, 0, 0, 0])

    return points, labels


def segment_with_timing(model, img_path, output_path, points, labels):
    """分割并记录时间"""
    start_time = time.time()

    # 预处理
    preprocess_start = time.time()
    roi_image, meta_info = preprocess_image(img_path)
    preprocess_time = time.time() - preprocess_start

    # 推理
    infer_start = time.time()
    roi_pred = infer_with_points(model, roi_image, points, labels, num_iterations=3)
    infer_time = time.time() - infer_start

    # 后处理
    postprocess_start = time.time()
    final_pred = postprocess_mask(roi_pred, meta_info)
    postprocess_time = time.time() - postprocess_start

    # 保存
    save_start = time.time()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    save_prediction(final_pred, output_path, meta_info)
    save_time = time.time() - save_start

    total_time = time.time() - start_time

    # 获取统计信息
    img_nib = nib.load(img_path)
    img_shape = img_nib.shape
    seg_voxels = np.sum(final_pred > 0)

    return {
        'preprocess_time': preprocess_time,
        'infer_time': infer_time,
        'postprocess_time': postprocess_time,
        'save_time': save_time,
        'total_time': total_time,
        'img_shape': img_shape,
        'seg_voxels': seg_voxels,
    }


def main():
    global results

    print("=" * 70)
    print("SAM-Med3D 脊柱MRI分割 - 带时间统计")
    print("=" * 70)

    # 加载模型
    model_load_start = time.time()
    model = get_model()
    model_load_time = time.time() - model_load_start
    print(f"\n模型加载时间: {model_load_time:.2f} 秒")

    nifti_dir = "/home/nyuair/junyi/SAM-Med3D/9hospital/nifti"
    output_dir = "/home/nyuair/junyi/SAM-Med3D/9hospital/segmentation_results_timed"

    points, labels = get_spine_prompt_points()

    for patient_dir in sorted(glob(os.path.join(nifti_dir, "*"))):
        patient = os.path.basename(patient_dir)
        print(f"\n{'='*60}")
        print(f"病人: {patient}")

        t2_files = glob(os.path.join(patient_dir, "*t2*sag*.nii.gz"))
        if not t2_files:
            t2_files = glob(os.path.join(patient_dir, "*sag*.nii.gz"))[:2]

        for img_path in t2_files[:2]:
            seq_name = os.path.basename(img_path).replace('.nii.gz', '')
            output_path = os.path.join(output_dir, patient, f"{seq_name}_seg.nii.gz")

            print(f"\n  处理: {seq_name}")

            try:
                timing = segment_with_timing(model, img_path, output_path, points, labels)

                print(f"    图像大小: {timing['img_shape']}")
                print(f"    预处理: {timing['preprocess_time']:.2f}s | 推理: {timing['infer_time']:.2f}s | 后处理: {timing['postprocess_time']:.2f}s")
                print(f"    总时间: {timing['total_time']:.2f}s | 分割体素: {timing['seg_voxels']:,}")

                results.append({
                    'patient': patient,
                    'sequence': seq_name,
                    'img_shape': timing['img_shape'],
                    'preprocess_time': timing['preprocess_time'],
                    'infer_time': timing['infer_time'],
                    'postprocess_time': timing['postprocess_time'],
                    'total_time': timing['total_time'],
                    'seg_voxels': timing['seg_voxels'],
                })

            except Exception as e:
                print(f"    错误: {e}")

    # 保存结果摘要
    write_readme(model_load_time)

    return results


def write_readme(model_load_time):
    """写入README文件"""
    readme_path = "/home/nyuair/junyi/SAM-Med3D/9hospital/README.md"

    total_time = sum(r['total_time'] for r in results)
    avg_time = total_time / len(results) if results else 0

    content = f"""# SAM-Med3D 脊柱MRI分割项目

## 项目概述

本项目使用 SAM-Med3D (Segment Anything Model for 3D Medical Images) 对脊柱MRI图像进行自动分割。

- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **模型**: SAM-Med3D-turbo
- **环境**: segmed (conda)

## 环境配置

### Conda 环境
```bash
conda activate segmed
```

### 主要依赖
- Python 3.10
- PyTorch 2.7.1 + CUDA 11.8
- nibabel, SimpleITK, pydicom (医学影像处理)
- monai, torchio, einops (深度学习医学影像)
- medim (SAM-Med3D 接口)

## 数据来源

| 病人 | DICOM文件 | 序列数 |
|------|-----------|--------|
| LIANG ZHI HAO | liang.zip | 2 |
| WU HONG BIN | wu.zip | 8 |
| ZHANG YONG SHENG | zhang.zip | 9 |

## 处理时间统计

### 模型加载
- **模型加载时间**: {model_load_time:.2f} 秒

### 单图处理时间

| 病人 | 序列 | 图像大小 | 预处理 | 推理 | 后处理 | 总时间 | 分割体素 |
|------|------|----------|--------|------|--------|--------|----------|
"""

    for r in results:
        shape_str = f"{r['img_shape'][0]}x{r['img_shape'][1]}x{r['img_shape'][2]}"
        content += f"| {r['patient'][:15]} | {r['sequence'][:25]} | {shape_str} | {r['preprocess_time']:.2f}s | {r['infer_time']:.2f}s | {r['postprocess_time']:.2f}s | {r['total_time']:.2f}s | {r['seg_voxels']:,} |\n"

    content += f"""
### 汇总统计
- **总处理图像数**: {len(results)}
- **总处理时间**: {total_time:.2f} 秒
- **平均每张时间**: {avg_time:.2f} 秒

## 输出文件

### 目录结构
```
9hospital/
├── nifti/                          # NIfTI格式原图
│   ├── LIANG_ZHI_HAO/
│   ├── WU_HONG_BIN/
│   └── ZHANG_YONG_SHENG/
├── segmentation_results_timed/     # 分割结果
│   ├── LIANG_ZHI_HAO/
│   ├── WU_HONG_BIN/
│   └── ZHANG_YONG_SHENG/
├── visualizations/                 # 可视化PNG
│   ├── LIANG_ZHI_HAO/
│   ├── WU_HONG_BIN/
│   └── ZHANG_YONG_SHENG/
├── run_segmentation.py            # 基础分割脚本
├── interactive_segment.py         # 交互式分割脚本
├── visualize_results.py           # 可视化脚本
└── README.md                      # 本文件
```

## 使用方法

### 1. 运行分割
```bash
conda activate segmed
cd /home/nyuair/junyi/SAM-Med3D/9hospital
python interactive_segment.py
```

### 2. 生成可视化
```bash
python visualize_results.py
```

### 3. 自定义提示点

编辑 `interactive_segment.py` 中的 `get_spine_prompt_points()` 函数来调整分割区域:

```python
def get_spine_prompt_points(image_shape=(128, 128, 128)):
    # 正样本点 (要分割的区域)
    points = [[64, 64, 32], [64, 64, 64], [64, 64, 96]]
    labels = [1, 1, 1]

    # 负样本点 (背景区域)
    points += [[10, 10, 64], [118, 118, 64]]
    labels += [0, 0]

    return points, labels
```

## 查看结果

### 使用 ITK-SNAP
```bash
itksnap -g nifti/LIANG_ZHI_HAO/t2_tse_dixon_sag_W.nii.gz \\
        -s segmentation_results_timed/LIANG_ZHI_HAO/t2_tse_dixon_sag_W_seg.nii.gz
```

### 使用 3D Slicer
1. 打开 3D Slicer
2. 导入 NIfTI 原图
3. 导入对应的分割结果作为 Segmentation

## 技术细节

### SAM-Med3D 模型
- 基于 Segment Anything Model (SAM) 架构
- 针对3D医学影像优化
- 支持点提示、框提示的交互式分割

### 处理流程
1. DICOM → NIfTI 转换
2. 重采样到统一分辨率 (1.5mm x 1.5mm x 1.5mm)
3. 裁剪/填充到 128x128x128
4. Z-score 归一化
5. SAM-Med3D 推理 (3次迭代)
6. 后处理映射回原始空间

## 参考资料

- [SAM-Med3D GitHub](https://github.com/uni-medical/SAM-Med3D)
- [SAM-Med3D Paper](https://arxiv.org/abs/2310.15161)
- [Hugging Face Model](https://huggingface.co/blueyo0/SAM-Med3D)
"""

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n\n已保存 README: {readme_path}")


if __name__ == "__main__":
    main()
