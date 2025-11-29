"""
SAM-Med3D 微调数据准备工具

将标注好的椎间盘数据转换为 SAM-Med3D 训练格式:
- 输入: NIfTI 图像 + 多标签掩码
- 输出: SAM-Med3D 格式的训练数据（每个椎间盘一个二值掩码）

数据目录结构要求:
    input_dir/
        Case01.nii.gz
        Case01_mask.nii.gz
        Case02.nii.gz
        Case02_mask.nii.gz
        ...

输出结构 (SAM-Med3D 格式):
    output_dir/
        ivd/
            mri_IVD/
                imagesTr/
                    Case01_L1L2.nii.gz
                    Case01_L2L3.nii.gz
                    ...
                labelsTr/
                    Case01_L1L2.nii.gz  (binary mask)
                    ...
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from glob import glob

import numpy as np
import nibabel as nib
import SimpleITK as sitk


# 椎间盘标签配置（与 ivd_degeneration_analysis 保持一致）
DISC_LABELS = {
    'L1L2': 3,
    'L2L3': 5,
    'L3L4': 7,
    'L4L5': 9,
    'L5S1': 11
}


def extract_disc_roi(
    image: np.ndarray,
    mask: np.ndarray,
    disc_label: int,
    padding: int = 16
) -> Tuple[np.ndarray, np.ndarray, Tuple[slice, slice, slice]]:
    """
    提取单个椎间盘的 ROI

    Args:
        image: 3D 图像数组
        mask: 多标签掩码数组
        disc_label: 目标椎间盘标签
        padding: ROI 周围的填充像素数

    Returns:
        (cropped_image, binary_mask, slices)
    """
    # 创建二值掩码
    binary_mask = (mask == disc_label).astype(np.uint8)

    # 检查是否存在该标签
    if binary_mask.sum() == 0:
        return None, None, None

    # 找到边界框
    coords = np.where(binary_mask > 0)
    z_min, z_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    x_min, x_max = coords[2].min(), coords[2].max()

    # 添加填充
    z_min = max(0, z_min - padding)
    z_max = min(image.shape[0], z_max + padding + 1)
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[1], y_max + padding + 1)
    x_min = max(0, x_min - padding)
    x_max = min(image.shape[2], x_max + padding + 1)

    slices = (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))

    cropped_image = image[slices]
    cropped_mask = binary_mask[slices]

    return cropped_image, cropped_mask, slices


def resample_to_size(
    image: sitk.Image,
    target_size: Tuple[int, int, int] = (128, 128, 128),
    is_label: bool = False
) -> sitk.Image:
    """
    重采样图像到目标尺寸

    Args:
        image: SimpleITK 图像
        target_size: 目标尺寸 (D, H, W)
        is_label: 是否为标签图像（使用最近邻插值）

    Returns:
        重采样后的图像
    """
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    # 计算新的 spacing
    new_spacing = [
        orig_sz * orig_sp / tgt_sz
        for orig_sz, orig_sp, tgt_sz in zip(original_size, original_spacing, target_size)
    ]

    # 重采样
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(image)


def prepare_single_case(
    image_path: str,
    mask_path: str,
    output_dir: str,
    case_name: str,
    target_size: Optional[Tuple[int, int, int]] = None,
    extract_roi: bool = True,
    roi_padding: int = 16
) -> List[str]:
    """
    处理单个病例

    Args:
        image_path: 图像路径
        mask_path: 掩码路径
        output_dir: 输出目录
        case_name: 病例名称
        target_size: 目标尺寸（如果为 None 则保持原始尺寸）
        extract_roi: 是否提取 ROI
        roi_padding: ROI 填充

    Returns:
        生成的文件路径列表
    """
    # 加载数据
    img_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)

    image = img_nii.get_fdata()
    mask = mask_nii.get_fdata()
    affine = img_nii.affine

    created_files = []

    # 创建输出目录
    images_dir = os.path.join(output_dir, "imagesTr")
    labels_dir = os.path.join(output_dir, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 处理每个椎间盘
    for disc_name, disc_label in DISC_LABELS.items():
        print(f"  处理 {disc_name} (标签 {disc_label})...")

        if extract_roi:
            cropped_image, cropped_mask, slices = extract_disc_roi(
                image, mask, disc_label, roi_padding
            )

            if cropped_image is None:
                print(f"    [SKIP] {disc_name} 不存在于掩码中")
                continue
        else:
            cropped_image = image
            cropped_mask = (mask == disc_label).astype(np.uint8)

            if cropped_mask.sum() == 0:
                print(f"    [SKIP] {disc_name} 不存在于掩码中")
                continue

        # 保存文件名
        output_name = f"{case_name}_{disc_name}"
        image_out_path = os.path.join(images_dir, f"{output_name}.nii.gz")
        label_out_path = os.path.join(labels_dir, f"{output_name}.nii.gz")

        if target_size:
            # 使用 SimpleITK 进行重采样
            img_sitk = sitk.GetImageFromArray(cropped_image.astype(np.float32))
            mask_sitk = sitk.GetImageFromArray(cropped_mask.astype(np.uint8))

            img_sitk = resample_to_size(img_sitk, target_size, is_label=False)
            mask_sitk = resample_to_size(mask_sitk, target_size, is_label=True)

            sitk.WriteImage(img_sitk, image_out_path)
            sitk.WriteImage(mask_sitk, label_out_path)
        else:
            # 直接保存
            out_img_nii = nib.Nifti1Image(cropped_image.astype(np.float32), affine)
            out_mask_nii = nib.Nifti1Image(cropped_mask.astype(np.uint8), affine)

            nib.save(out_img_nii, image_out_path)
            nib.save(out_mask_nii, label_out_path)

        created_files.append(image_out_path)
        print(f"    [OK] 保存到 {output_name}")

    return created_files


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    target_size: Optional[Tuple[int, int, int]] = (128, 128, 128),
    extract_roi: bool = True,
    roi_padding: int = 16
) -> Dict[str, List[str]]:
    """
    准备整个数据集

    Args:
        input_dir: 输入目录（包含 NIfTI 图像和掩码）
        output_dir: 输出目录
        target_size: 目标尺寸
        extract_roi: 是否提取 ROI
        roi_padding: ROI 填充

    Returns:
        处理结果统计
    """
    # 查找所有图像文件
    image_files = glob(os.path.join(input_dir, "*.nii.gz"))
    image_files = [f for f in image_files if not f.endswith("_mask.nii.gz")]

    if not image_files:
        # 尝试查找 .nii 文件
        image_files = glob(os.path.join(input_dir, "*.nii"))
        image_files = [f for f in image_files if not f.endswith("_mask.nii")]

    print(f"找到 {len(image_files)} 个病例")

    results = {
        'success': [],
        'failed': [],
        'total_discs': 0
    }

    # SAM-Med3D 数据目录结构
    sam_data_dir = os.path.join(output_dir, "ivd", "mri_IVD")
    os.makedirs(sam_data_dir, exist_ok=True)

    for image_path in image_files:
        case_name = Path(image_path).stem.replace('.nii', '')

        # 查找对应的掩码
        mask_path = image_path.replace('.nii.gz', '_mask.nii.gz').replace('.nii', '_mask.nii')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(
                os.path.dirname(image_path),
                case_name + "_mask.nii.gz"
            )

        if not os.path.exists(mask_path):
            print(f"[WARNING] 未找到 {case_name} 的掩码文件，跳过")
            results['failed'].append(case_name)
            continue

        print(f"\n处理病例: {case_name}")

        try:
            created_files = prepare_single_case(
                image_path,
                mask_path,
                sam_data_dir,
                case_name,
                target_size,
                extract_roi,
                roi_padding
            )
            results['success'].append(case_name)
            results['total_discs'] += len(created_files)
        except Exception as e:
            print(f"[ERROR] 处理 {case_name} 失败: {e}")
            results['failed'].append(case_name)

    # 生成数据路径配置文件
    data_paths_content = f'''"""
SAM-Med3D 训练数据路径配置
自动生成于椎间盘分割数据准备脚本
"""
import os.path as osp
from glob import glob

# 椎间盘分割数据路径
PROJ_DIR = osp.dirname(osp.dirname(__file__))
img_datas = [
    "{sam_data_dir}",
]
'''

    config_path = os.path.join(output_dir, "data_paths_ivd.py")
    with open(config_path, 'w') as f:
        f.write(data_paths_content)

    print(f"\n{'='*60}")
    print("数据准备完成!")
    print(f"成功处理: {len(results['success'])} 个病例")
    print(f"失败: {len(results['failed'])} 个病例")
    print(f"总计椎间盘数: {results['total_discs']}")
    print(f"数据目录: {sam_data_dir}")
    print(f"配置文件: {config_path}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SAM-Med3D 微调数据准备工具")
    parser.add_argument("input_dir", help="输入目录（包含 NIfTI 图像和掩码）")
    parser.add_argument("output_dir", help="输出目录")
    parser.add_argument("--target-size", type=int, nargs=3, default=[128, 128, 128],
                        help="目标尺寸 (D H W)，默认 128 128 128")
    parser.add_argument("--no-resize", action="store_true",
                        help="不进行尺寸调整")
    parser.add_argument("--no-roi", action="store_true",
                        help="不提取 ROI，使用完整图像")
    parser.add_argument("--roi-padding", type=int, default=16,
                        help="ROI 填充像素数")

    args = parser.parse_args()

    target_size = None if args.no_resize else tuple(args.target_size)

    prepare_dataset(
        args.input_dir,
        args.output_dir,
        target_size=target_size,
        extract_roi=not args.no_roi,
        roi_padding=args.roi_padding
    )


if __name__ == "__main__":
    main()
