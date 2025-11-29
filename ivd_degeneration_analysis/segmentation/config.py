"""
椎间盘分割模块配置文件

所有路径和参数配置集中在此处管理
"""

import os

# ============== 路径配置 ==============

# SAM-Med3D 项目路径
SAM_MED3D_PATH = "/home/nyuair/junyi/SAM-Med3D"

# 原始 DICOM 数据目录（包含 liang.zip, wu.zip, zhang.zip）
DICOM_DATA_DIR = "/home/nyuair/junyi/SAM-Med3D/9hospital"

# 输出目录
OUTPUT_BASE_DIR = "/home/nyuair/junyi/ivd_degeneration_analysis/segmentation/output"

# NIfTI 转换输出目录
NIFTI_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "nifti")

# 分割掩码输出目录
MASK_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "masks")

# 微调数据目录
FINETUNE_DATA_DIR = os.path.join(OUTPUT_BASE_DIR, "finetune_data")

# 微调模型输出目录
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "models")

# SAM-Med3D 预训练权重路径（需要下载）
SAM_MED3D_CHECKPOINT = os.path.join(SAM_MED3D_PATH, "ckpt", "sam_med3d_turbo.pth")


# ============== 椎间盘标签配置 ==============

# 与 ivd_degeneration_analysis 保持一致
DISC_LABELS = {
    'L1-L2': {'disc': 3, 'upper': 2, 'lower': 4},
    'L2-L3': {'disc': 5, 'upper': 4, 'lower': 6},
    'L3-L4': {'disc': 7, 'upper': 6, 'lower': 8},
    'L4-L5': {'disc': 9, 'upper': 8, 'lower': 10},
    'L5-S1': {'disc': 11, 'upper': 10, 'lower': 12}
}

# 硬膜囊标签
DURAL_SAC_LABEL = 20


# ============== 模型参数 ==============

# 图像尺寸（SAM-Med3D 默认）
IMG_SIZE = 128

# 每个样本的点击点数量
NUM_CLICKS = 5

# 训练参数默认值
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 2
DEFAULT_LR = 1e-4


# ============== 病例信息 ==============

# 当前可用的病例
AVAILABLE_CASES = {
    'liang': os.path.join(DICOM_DATA_DIR, 'liang.zip'),
    'wu': os.path.join(DICOM_DATA_DIR, 'wu.zip'),
    'zhang': os.path.join(DICOM_DATA_DIR, 'zhang.zip'),
}


def ensure_dirs():
    """确保所有输出目录存在"""
    dirs = [
        OUTPUT_BASE_DIR,
        NIFTI_OUTPUT_DIR,
        MASK_OUTPUT_DIR,
        FINETUNE_DATA_DIR,
        MODEL_OUTPUT_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_case_paths(case_name: str) -> dict:
    """
    获取指定病例的所有相关路径

    Args:
        case_name: 病例名称 (liang, wu, zhang)

    Returns:
        包含各种路径的字典
    """
    return {
        'dicom_zip': AVAILABLE_CASES.get(case_name),
        'nifti': os.path.join(NIFTI_OUTPUT_DIR, f"{case_name}.nii.gz"),
        'mask': os.path.join(MASK_OUTPUT_DIR, f"{case_name}_mask.nii.gz"),
    }
