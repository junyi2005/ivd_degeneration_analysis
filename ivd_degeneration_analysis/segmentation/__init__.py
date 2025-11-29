"""
椎间盘分割模块

提供两种分割方案:
1. 交互式分割 (interactive_segment.py) - 适合少量病例
2. 微调分割 (finetune_sam_med3d.py) - 适合大量病例

工具:
- dicom_to_nifti.py: DICOM 转 NIfTI
- prepare_finetune_data.py: 准备微调数据
- run_segmentation_pipeline.py: 完整工作流
"""

from .dicom_to_nifti import convert_dicom_to_nifti, batch_convert

__all__ = [
    'convert_dicom_to_nifti',
    'batch_convert',
]
