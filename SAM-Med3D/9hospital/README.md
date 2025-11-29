# SAM-Med3D 脊柱MRI分割项目

## 项目概述

本项目使用 SAM-Med3D (Segment Anything Model for 3D Medical Images) 对脊柱MRI图像进行自动分割。

- **生成时间**: 2025-11-30 01:12:46
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
- **模型加载时间**: 3.07 秒

### 单图处理时间

| 病人 | 序列 | 图像大小 | 预处理 | 推理 | 后处理 | 总时间 | 分割体素 |
|------|------|----------|--------|------|--------|--------|----------|
| LIANG_ZHI_HAO | t2_tse_dixon_sag_F | 640x640x15 | 1.02s | 1.37s | 0.16s | 2.56s | 25,977 |
| LIANG_ZHI_HAO | t2_tse_dixon_sag_W | 640x640x15 | 1.11s | 0.17s | 0.15s | 1.44s | 15,878 |
| WU_HONG_BIN | t2_tse_dixon_sag_F | 640x640x15 | 0.85s | 0.19s | 0.17s | 1.23s | 15,055 |
| WU_HONG_BIN | t2_tse_dixon_sag_in | 640x640x15 | 1.06s | 0.18s | 0.12s | 1.37s | 14,531 |
| ZHANG_YONG_SHEN | SMR240d:t2_fse_sag_384_DN | 368x368x13 | 0.44s | 0.17s | 0.09s | 0.70s | 2,602 |
| ZHANG_YONG_SHEN | t2_fse_sag_384_DNEl | 368x368x13 | 0.56s | 0.20s | 0.14s | 0.90s | 2,635 |

### 汇总统计
- **总处理图像数**: 6
- **总处理时间**: 8.21 秒
- **平均每张时间**: 1.37 秒

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
itksnap -g nifti/LIANG_ZHI_HAO/t2_tse_dixon_sag_W.nii.gz \
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
