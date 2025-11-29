"""
椎间盘分割完整工作流

此脚本整合了从 DICOM 转换到 SAM-Med3D 分割的完整流程:
1. DICOM ZIP 解压并转换为 NIfTI
2. 使用 SAM-Med3D 进行交互式或自动分割
3. 生成与 ivd_degeneration_analysis 兼容的掩码格式

使用示例:
    # 交互式分割单个病例
    python run_segmentation_pipeline.py --input liang.zip --interactive

    # 批量处理多个病例
    python run_segmentation_pipeline.py --input-dir ./dicom_zips --batch
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dicom_to_nifti import convert_dicom_to_nifti, batch_convert


def setup_environment():
    """设置环境和依赖检查"""
    required_packages = [
        'SimpleITK',
        'nibabel',
        'torch',
        'matplotlib',
        'numpy'
    ]

    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[WARNING] 缺少以下依赖包: {missing}")
        print("请运行: pip install " + " ".join(missing))

    # 检查 medim
    try:
        import medim
    except ImportError:
        print("[WARNING] 缺少 medim 包，请运行: pip install medim")


def run_interactive_segmentation(
    nifti_path: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None
):
    """运行交互式分割"""
    from interactive_segment import SAMMed3DSegmenter, InteractiveSegmentationGUI

    print(f"\n{'='*60}")
    print(f"开始交互式分割: {nifti_path}")
    print(f"{'='*60}\n")

    segmenter = SAMMed3DSegmenter(checkpoint_path=checkpoint_path)
    gui = InteractiveSegmentationGUI(nifti_path, segmenter, output_dir)
    gui.run()


def run_auto_segmentation(
    nifti_path: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    use_center_points: bool = True
):
    """
    运行自动分割（使用图像中心作为提示点）

    这是一个简化版本，实际使用中可能需要更智能的提示点选择策略
    """
    import numpy as np
    import nibabel as nib
    from interactive_segment import SAMMed3DSegmenter, PromptPoint

    print(f"\n{'='*60}")
    print(f"开始自动分割: {nifti_path}")
    print(f"{'='*60}\n")

    # 加载图像
    nii = nib.load(nifti_path)
    image = nii.get_fdata()

    # 初始化分割器
    segmenter = SAMMed3DSegmenter(checkpoint_path=checkpoint_path)

    # 椎间盘标签配置
    disc_labels = {
        'L1-L2': 3,
        'L2-L3': 5,
        'L3-L4': 7,
        'L4-L5': 9,
        'L5-S1': 11
    }

    # 创建合并掩码
    combined_mask = np.zeros(image.shape, dtype=np.uint8)

    if use_center_points:
        # 使用图像中心区域作为初始提示点
        # 这是一个简化策略，实际使用可能需要更复杂的定位算法
        center_z = image.shape[0] // 2
        center_y = image.shape[1] // 2
        center_x = image.shape[2] // 2

        # 估算椎间盘位置（假设矢状位 MRI，椎间盘在中线附近）
        # 这里需要根据实际图像调整
        disc_positions = {
            'L1-L2': (center_z - 30, center_y, center_x),
            'L2-L3': (center_z - 15, center_y, center_x),
            'L3-L4': (center_z, center_y, center_x),
            'L4-L5': (center_z + 15, center_y, center_x),
            'L5-S1': (center_z + 30, center_y, center_x),
        }

        for disc_name, (z, y, x) in disc_positions.items():
            # 确保坐标在有效范围内
            z = max(0, min(z, image.shape[0] - 1))
            y = max(0, min(y, image.shape[1] - 1))
            x = max(0, min(x, image.shape[2] - 1))

            print(f"[INFO] 分割 {disc_name} 使用提示点: ({x}, {y}, {z})")

            prompt_points = [PromptPoint(x=x, y=y, z=z, label=1)]

            try:
                mask = segmenter.segment_with_prompts(image, prompt_points)
                combined_mask[mask > 0] = disc_labels[disc_name]
                print(f"[SUCCESS] {disc_name} 分割完成")
            except Exception as e:
                print(f"[WARNING] {disc_name} 分割失败: {e}")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    case_name = Path(nifti_path).stem.replace('.nii', '')
    output_path = os.path.join(output_dir, f"{case_name}_mask.nii.gz")

    mask_nii = nib.Nifti1Image(combined_mask, nii.affine)
    nib.save(mask_nii, output_path)
    print(f"\n[SUCCESS] 分割掩码已保存到: {output_path}")

    return output_path


def process_single_case(
    input_path: str,
    output_dir: str,
    interactive: bool = True,
    checkpoint_path: Optional[str] = None,
    skip_conversion: bool = False
):
    """处理单个病例"""
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 转换 DICOM 到 NIfTI（如果需要）
    if input_path.endswith('.nii') or input_path.endswith('.nii.gz'):
        nifti_path = input_path
        print(f"[INFO] 输入已是 NIfTI 格式: {nifti_path}")
    elif skip_conversion:
        print("[ERROR] --skip-conversion 仅适用于 NIfTI 输入")
        return
    else:
        case_name = Path(input_path).stem
        nifti_path = os.path.join(output_dir, "nifti", f"{case_name}.nii.gz")

        print(f"[Step 1] 转换 DICOM 到 NIfTI...")
        convert_dicom_to_nifti(input_path, nifti_path)

    # Step 2: 分割
    seg_output_dir = os.path.join(output_dir, "masks")
    print(f"\n[Step 2] 执行分割...")

    if interactive:
        run_interactive_segmentation(nifti_path, seg_output_dir, checkpoint_path)
    else:
        run_auto_segmentation(nifti_path, seg_output_dir, checkpoint_path)

    print(f"\n{'='*60}")
    print("处理完成!")
    print(f"NIfTI 图像: {nifti_path}")
    print(f"分割掩码: {seg_output_dir}/")
    print(f"{'='*60}")


def process_batch(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.zip",
    interactive: bool = False,
    checkpoint_path: Optional[str] = None
):
    """批量处理多个病例"""
    from glob import glob

    # Step 1: 批量转换 DICOM
    nifti_dir = os.path.join(output_dir, "nifti")
    print(f"[Step 1] 批量转换 DICOM 到 NIfTI...")
    nifti_files = batch_convert(input_dir, nifti_dir, pattern)

    if not nifti_files:
        print("[ERROR] 没有成功转换的文件")
        return

    # Step 2: 批量分割
    seg_output_dir = os.path.join(output_dir, "masks")
    print(f"\n[Step 2] 批量分割...")

    for nifti_path in nifti_files:
        if interactive:
            # 交互模式逐个处理
            run_interactive_segmentation(nifti_path, seg_output_dir, checkpoint_path)
        else:
            # 自动模式
            try:
                run_auto_segmentation(nifti_path, seg_output_dir, checkpoint_path)
            except Exception as e:
                print(f"[ERROR] 分割 {nifti_path} 失败: {e}")

    print(f"\n{'='*60}")
    print(f"批量处理完成! 共处理 {len(nifti_files)} 个文件")
    print(f"NIfTI 目录: {nifti_dir}")
    print(f"掩码目录: {seg_output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="椎间盘分割完整工作流",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 交互式分割单个 DICOM ZIP
  python run_segmentation_pipeline.py --input liang.zip --interactive

  # 交互式分割已有的 NIfTI 文件
  python run_segmentation_pipeline.py --input case01.nii.gz --interactive

  # 自动分割（使用中心点作为提示）
  python run_segmentation_pipeline.py --input liang.zip

  # 批量处理多个 ZIP 文件
  python run_segmentation_pipeline.py --input-dir ./dicom_zips --batch

  # 使用自定义模型权重
  python run_segmentation_pipeline.py --input liang.zip --checkpoint /path/to/model.pth
        """
    )

    parser.add_argument("--input", "-i", help="单个输入文件（DICOM 目录、ZIP 或 NIfTI）")
    parser.add_argument("--input-dir", help="批量处理的输入目录")
    parser.add_argument("--output-dir", "-o", default="./ivd_segmentation_output",
                        help="输出目录")
    parser.add_argument("--interactive", action="store_true",
                        help="使用交互式分割模式")
    parser.add_argument("--batch", action="store_true",
                        help="批量处理模式")
    parser.add_argument("--pattern", default="*.zip",
                        help="批量模式的文件匹配模式")
    parser.add_argument("--checkpoint", "-c",
                        help="SAM-Med3D 模型权重路径")
    parser.add_argument("--skip-conversion", action="store_true",
                        help="跳过 DICOM 转换（仅适用于 NIfTI 输入）")

    args = parser.parse_args()

    # 环境检查
    setup_environment()

    if args.batch:
        if not args.input_dir:
            print("[ERROR] 批量模式需要 --input-dir 参数")
            return
        process_batch(
            args.input_dir,
            args.output_dir,
            args.pattern,
            args.interactive,
            args.checkpoint
        )
    elif args.input:
        process_single_case(
            args.input,
            args.output_dir,
            args.interactive,
            args.checkpoint,
            args.skip_conversion
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
