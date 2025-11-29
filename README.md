# IVD Degeneration Analysis & SAM-Med3D 项目

## 项目概述

本项目用于椎间盘退变 (IVD Degeneration) 的医学图像分析，结合 SAM-Med3D 进行 3D 医学图像分割。

## 更新内容

### ivd_degeneration_analysis

相对于原始代码，主要更新在 **`segmentation/`** 文件夹：

- `segmentation/` - 新增的分割模块
  - `run_segmentation_pipeline.py` - 完整分割流程
  - `dicom_to_nifti.py` - DICOM 转 NIfTI
  - `interactive_segment.py` - SAM-Med3D 交互式分割
  - `finetune_sam_med3d.py` - 模型微调脚本
  - `config.py` - 配置文件

### SAM-Med3D

相对于原始 uni-medical/SAM-Med3D 仓库，主要更新在 **`9hospital/`** 文件夹：

- `9hospital/` - 医院数据处理与分割结果
  - `run_segmentation.py` - 分割脚本
  - `interactive_segment.py` - 改进的多点提示分割
  - `run_with_timing.py` - 带计时统计的分割
  - `visualize_results.py` - 可视化生成
  - `segmentation_results*/` - 分割结果 (NIfTI 格式)
  - `visualizations/` - 可视化 PNG 图片
  - `README.md` - 详细处理说明与计时统计
