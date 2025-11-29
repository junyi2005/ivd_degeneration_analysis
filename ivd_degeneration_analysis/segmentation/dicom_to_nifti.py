"""
DICOM 转 NIfTI 工具
将医院导出的 DICOM 序列转换为 NIfTI 格式，供 SAM-Med3D 使用
"""

import os
import argparse
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import SimpleITK as sitk


def extract_zip(zip_path: str, extract_to: str) -> str:
    """解压 ZIP 文件到指定目录"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to


def find_dicom_series(directory: str) -> List[str]:
    """
    查找目录中的所有 DICOM 系列
    返回系列 ID 列表
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(directory)
    return list(series_ids)


def load_dicom_series(directory: str, series_id: Optional[str] = None) -> Tuple[sitk.Image, dict]:
    """
    加载 DICOM 序列为 SimpleITK Image

    Args:
        directory: DICOM 文件所在目录
        series_id: 指定的系列 ID，如果为 None 则自动选择第一个

    Returns:
        (sitk.Image, metadata_dict)
    """
    reader = sitk.ImageSeriesReader()

    if series_id is None:
        series_ids = reader.GetGDCMSeriesIDs(directory)
        if not series_ids:
            raise ValueError(f"在 {directory} 中未找到 DICOM 系列")
        series_id = series_ids[0]
        if len(series_ids) > 1:
            print(f"[INFO] 发现 {len(series_ids)} 个 DICOM 系列，使用第一个: {series_id}")

    dicom_files = reader.GetGDCMSeriesFileNames(directory, series_id)
    if not dicom_files:
        raise ValueError(f"系列 {series_id} 中未找到 DICOM 文件")

    print(f"[INFO] 加载 {len(dicom_files)} 个 DICOM 文件...")

    reader.SetFileNames(dicom_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    image = reader.Execute()

    # 提取元数据
    metadata = {}
    if len(dicom_files) > 0:
        for key in reader.GetMetaDataKeys(0):
            try:
                metadata[key] = reader.GetMetaData(0, key)
            except:
                pass

    return image, metadata


def convert_dicom_to_nifti(
    input_path: str,
    output_path: str,
    series_id: Optional[str] = None,
    is_zip: bool = False
) -> str:
    """
    将 DICOM 转换为 NIfTI 格式

    Args:
        input_path: DICOM 目录路径或 ZIP 文件路径
        output_path: 输出的 NIfTI 文件路径
        series_id: 指定的 DICOM 系列 ID
        is_zip: 输入是否为 ZIP 文件

    Returns:
        输出文件路径
    """
    temp_dir = None

    try:
        if is_zip or input_path.endswith('.zip'):
            temp_dir = tempfile.mkdtemp()
            print(f"[INFO] 解压 ZIP 文件到临时目录: {temp_dir}")
            extract_zip(input_path, temp_dir)
            dicom_dir = temp_dir

            # 查找包含 DICOM 文件的子目录
            for root, dirs, files in os.walk(temp_dir):
                dcm_files = [f for f in files if f.endswith('.dcm') or f.startswith('MR.')]
                if dcm_files:
                    dicom_dir = root
                    break
        else:
            dicom_dir = input_path

        # 加载 DICOM 序列
        image, metadata = load_dicom_series(dicom_dir, series_id)

        # 打印图像信息
        print(f"[INFO] 图像尺寸: {image.GetSize()}")
        print(f"[INFO] 体素间距: {image.GetSpacing()}")
        print(f"[INFO] 图像方向: {image.GetDirection()}")
        print(f"[INFO] 图像原点: {image.GetOrigin()}")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # 保存为 NIfTI
        sitk.WriteImage(image, output_path)
        print(f"[SUCCESS] 已保存到: {output_path}")

        return output_path

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def batch_convert(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.zip"
) -> List[str]:
    """
    批量转换多个 DICOM ZIP 文件

    Args:
        input_dir: 包含 ZIP 文件的输入目录
        output_dir: NIfTI 输出目录
        pattern: 文件匹配模式

    Returns:
        成功转换的文件路径列表
    """
    from glob import glob

    os.makedirs(output_dir, exist_ok=True)

    zip_files = glob(os.path.join(input_dir, pattern))
    print(f"[INFO] 找到 {len(zip_files)} 个文件待转换")

    converted = []
    for zip_path in zip_files:
        case_name = Path(zip_path).stem
        output_path = os.path.join(output_dir, f"{case_name}.nii.gz")

        try:
            convert_dicom_to_nifti(zip_path, output_path, is_zip=True)
            converted.append(output_path)
        except Exception as e:
            print(f"[ERROR] 转换 {zip_path} 失败: {e}")

    print(f"[INFO] 成功转换 {len(converted)}/{len(zip_files)} 个文件")
    return converted


def main():
    parser = argparse.ArgumentParser(description="DICOM 转 NIfTI 工具")
    parser.add_argument("input", help="输入路径（DICOM 目录或 ZIP 文件）")
    parser.add_argument("output", help="输出 NIfTI 文件路径")
    parser.add_argument("--series-id", help="指定 DICOM 系列 ID")
    parser.add_argument("--batch", action="store_true", help="批量转换模式")
    parser.add_argument("--pattern", default="*.zip", help="批量模式下的文件匹配模式")

    args = parser.parse_args()

    if args.batch:
        batch_convert(args.input, args.output, args.pattern)
    else:
        convert_dicom_to_nifti(args.input, args.output, args.series_id)


if __name__ == "__main__":
    main()
