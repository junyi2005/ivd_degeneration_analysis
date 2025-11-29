#!/usr/bin/env python3
"""
椎间盘分割快速启动脚本

简化的命令行界面，方便快速使用

使用示例:
    # 查看帮助
    python quick_start.py --help

    # 转换所有 DICOM 数据
    python quick_start.py convert

    # 交互式分割指定病例
    python quick_start.py segment liang

    # 批量转换并分割
    python quick_start.py all
"""

import os
import sys
import argparse

# 确保能导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    AVAILABLE_CASES, NIFTI_OUTPUT_DIR, MASK_OUTPUT_DIR,
    SAM_MED3D_CHECKPOINT, ensure_dirs
)


def convert_all():
    """转换所有 DICOM 数据到 NIfTI"""
    from dicom_to_nifti import convert_dicom_to_nifti

    ensure_dirs()

    print("\n" + "=" * 60)
    print("开始转换 DICOM 数据到 NIfTI 格式")
    print("=" * 60 + "\n")

    for case_name, zip_path in AVAILABLE_CASES.items():
        if not os.path.exists(zip_path):
            print(f"[SKIP] {case_name}: 文件不存在 {zip_path}")
            continue

        output_path = os.path.join(NIFTI_OUTPUT_DIR, f"{case_name}.nii.gz")

        if os.path.exists(output_path):
            print(f"[SKIP] {case_name}: 已存在 {output_path}")
            continue

        print(f"\n[处理] {case_name}...")
        try:
            convert_dicom_to_nifti(zip_path, output_path, is_zip=True)
            print(f"[成功] 保存到 {output_path}")
        except Exception as e:
            print(f"[失败] {case_name}: {e}")

    print("\n转换完成！")
    print(f"输出目录: {NIFTI_OUTPUT_DIR}")


def convert_single(case_name: str):
    """转换单个病例"""
    from dicom_to_nifti import convert_dicom_to_nifti

    ensure_dirs()

    if case_name not in AVAILABLE_CASES:
        print(f"[错误] 未知病例: {case_name}")
        print(f"可用病例: {list(AVAILABLE_CASES.keys())}")
        return None

    zip_path = AVAILABLE_CASES[case_name]
    output_path = os.path.join(NIFTI_OUTPUT_DIR, f"{case_name}.nii.gz")

    if not os.path.exists(zip_path):
        print(f"[错误] 文件不存在: {zip_path}")
        return None

    print(f"转换 {case_name}...")
    convert_dicom_to_nifti(zip_path, output_path, is_zip=True)
    print(f"保存到: {output_path}")

    return output_path


def segment_interactive(case_name: str, checkpoint: str = None):
    """交互式分割"""
    from interactive_segment import SAMMed3DSegmenter, InteractiveSegmentationGUI

    ensure_dirs()

    # 检查 NIfTI 文件
    nifti_path = os.path.join(NIFTI_OUTPUT_DIR, f"{case_name}.nii.gz")

    if not os.path.exists(nifti_path):
        print(f"[INFO] NIfTI 文件不存在，先进行转换...")
        nifti_path = convert_single(case_name)
        if nifti_path is None:
            return

    print(f"\n启动交互式分割: {case_name}")
    print(f"图像文件: {nifti_path}")
    print(f"输出目录: {MASK_OUTPUT_DIR}\n")

    # 使用模型
    ckpt = checkpoint or SAM_MED3D_CHECKPOINT
    if ckpt and not os.path.exists(ckpt):
        print(f"[警告] 模型权重不存在: {ckpt}")
        print("[INFO] 将尝试从 HuggingFace 下载...")
        ckpt = None

    segmenter = SAMMed3DSegmenter(checkpoint_path=ckpt)
    gui = InteractiveSegmentationGUI(nifti_path, segmenter, MASK_OUTPUT_DIR)
    gui.run()


def show_status():
    """显示当前数据状态"""
    print("\n" + "=" * 60)
    print("椎间盘分割项目状态")
    print("=" * 60)

    print("\n【原始 DICOM 数据】")
    for case_name, zip_path in AVAILABLE_CASES.items():
        status = "✓ 存在" if os.path.exists(zip_path) else "✗ 不存在"
        size = ""
        if os.path.exists(zip_path):
            size_mb = os.path.getsize(zip_path) / (1024 * 1024)
            size = f" ({size_mb:.1f} MB)"
        print(f"  {case_name}: {status}{size}")

    print("\n【NIfTI 转换结果】")
    for case_name in AVAILABLE_CASES.keys():
        nifti_path = os.path.join(NIFTI_OUTPUT_DIR, f"{case_name}.nii.gz")
        status = "✓ 已转换" if os.path.exists(nifti_path) else "✗ 未转换"
        print(f"  {case_name}: {status}")

    print("\n【分割掩码】")
    for case_name in AVAILABLE_CASES.keys():
        mask_path = os.path.join(MASK_OUTPUT_DIR, f"{case_name}_mask.nii.gz")
        if os.path.exists(mask_path):
            print(f"  {case_name}: ✓ 已分割")
        elif os.path.exists(MASK_OUTPUT_DIR):
            # 检查是否有单独的椎间盘掩码
            disc_masks = [f for f in os.listdir(MASK_OUTPUT_DIR)
                          if f.startswith(case_name) and f.endswith("_mask.nii.gz")]
            if disc_masks:
                print(f"  {case_name}: ✓ 部分分割 ({len(disc_masks)} 个椎间盘)")
            else:
                print(f"  {case_name}: ✗ 未分割")
        else:
            print(f"  {case_name}: ✗ 未分割")

    print("\n【模型权重】")
    if os.path.exists(SAM_MED3D_CHECKPOINT):
        size_mb = os.path.getsize(SAM_MED3D_CHECKPOINT) / (1024 * 1024)
        print(f"  SAM-Med3D-turbo: ✓ 存在 ({size_mb:.1f} MB)")
    else:
        print(f"  SAM-Med3D-turbo: ✗ 未下载")
        print(f"  下载地址: https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="椎间盘分割快速启动工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python quick_start.py status              # 查看数据状态
  python quick_start.py convert             # 转换所有 DICOM 到 NIfTI
  python quick_start.py convert liang       # 只转换 liang 病例
  python quick_start.py segment liang       # 交互式分割 liang 病例
  python quick_start.py all                 # 转换所有并逐个分割

可用病例: liang, wu, zhang
        """
    )

    parser.add_argument("action", choices=["convert", "segment", "status", "all"],
                        help="执行的操作")
    parser.add_argument("case", nargs="?", default=None,
                        help="病例名称 (liang/wu/zhang)")
    parser.add_argument("--checkpoint", "-c", default=None,
                        help="模型权重路径")

    args = parser.parse_args()

    if args.action == "status":
        show_status()

    elif args.action == "convert":
        if args.case:
            convert_single(args.case)
        else:
            convert_all()

    elif args.action == "segment":
        if not args.case:
            print("[错误] segment 操作需要指定病例名称")
            print("用法: python quick_start.py segment <case_name>")
            print("可用病例: liang, wu, zhang")
            return
        segment_interactive(args.case, args.checkpoint)

    elif args.action == "all":
        convert_all()
        print("\n开始逐个分割...")
        for case_name in AVAILABLE_CASES.keys():
            print(f"\n{'='*40}")
            print(f"分割病例: {case_name}")
            print(f"{'='*40}")
            segment_interactive(case_name, args.checkpoint)


if __name__ == "__main__":
    main()
