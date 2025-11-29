#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM-Med3D 脊柱MRI分割脚本
使用手动提示点进行分割，无需ground-truth标签
"""

import os
import sys
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import torchio as tio
from glob import glob

# 添加SAM-Med3D路径
sys.path.insert(0, '/home/nyuair/junyi/SAM-Med3D')

import medim

def load_model(ckpt_path=None):
    """加载SAM-Med3D模型"""
    if ckpt_path is None:
        ckpt_path = "/home/nyuair/junyi/SAM-Med3D/ckpt/sam_med3d_turbo.pth"

    print(f"加载模型: {ckpt_path}")
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
    return model


def preprocess_image(img_path, crop_size=128, target_spacing=(1.5, 1.5, 1.5)):
    """预处理图像"""
    # 读取图像
    sitk_image = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(sitk_image)  # Z, Y, X

    meta_info = {
        "sitk_image_object": sitk_image,
        "sitk_origin": sitk_image.GetOrigin(),
        "sitk_direction": sitk_image.GetDirection(),
        "sitk_spacing": sitk_image.GetSpacing(),
        "original_numpy_shape": img_arr.shape,
    }

    # 创建TorchIO subject (使用图像本身作为假标签)
    subject = tio.Subject(
        image=tio.ScalarImage(img_path),
    )

    meta_info["original_subject_affine"] = subject.image.affine.copy()
    meta_info["original_subject_spatial_shape"] = subject.image.spatial_shape

    # 重采样
    resampler = tio.Resample(target=target_spacing)
    subject_resampled = resampler(subject)

    # 规范化方向
    transform_canonical = tio.ToCanonical()
    subject_canonical = transform_canonical(subject_resampled)

    meta_info["canonical_subject_shape"] = subject_canonical.spatial_shape
    meta_info["canonical_subject_affine"] = subject_canonical.image.affine.copy()

    # 裁剪或填充到目标大小
    crop_transform = tio.CropOrPad(target_shape=(crop_size, crop_size, crop_size))
    padding_params, cropping_params = crop_transform._compute_center_crop_or_pad(subject_canonical)
    subject_cropped = crop_transform(subject_canonical)

    meta_info["padding_params_functional"] = padding_params
    meta_info["cropping_params_functional"] = cropping_params
    meta_info["roi_subject_affine"] = subject_cropped.image.affine.copy()

    # 归一化
    img3D_roi = subject_cropped.image.data.clone().detach()
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    img3D_roi = norm_transform(img3D_roi.squeeze(dim=1))
    img3D_roi = img3D_roi.unsqueeze(dim=1)

    # 确保5D
    if img3D_roi.ndim == 4:
        img3D_roi = img3D_roi.unsqueeze(0)

    return img3D_roi, meta_info


def infer_with_point(model, roi_image, point_coords, point_labels):
    """使用指定提示点进行推理"""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        input_tensor = roi_image.to(device)
        image_embeddings = model.image_encoder(input_tensor)

        # 准备提示点
        points_coords = torch.tensor(point_coords, dtype=torch.float32, device=device).reshape(1, -1, 3)
        points_labels = torch.tensor(point_labels, dtype=torch.int64, device=device).reshape(1, -1)

        # 初始化mask
        prev_low_res_mask = torch.zeros(1, 1,
                                        roi_image.shape[2] // 4,
                                        roi_image.shape[3] // 4,
                                        roi_image.shape[4] // 4,
                                        device=device, dtype=torch.float)

        # 编码提示
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=[points_coords, points_labels],
            boxes=None,
            masks=prev_low_res_mask,
        )

        # 解码mask
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )

        # 上采样到原始大小
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
    """后处理：将预测结果映射回原始图像空间"""
    roi_pred_tensor = torch.from_numpy(roi_pred_numpy.astype(np.float32)).unsqueeze(0)

    pred_label_map_roi_space = tio.LabelMap(
        tensor=roi_pred_tensor,
        affine=meta_info["roi_subject_affine"]
    )

    reference_tensor_shape = (1, *meta_info["original_subject_spatial_shape"])
    reference_image_original_space = tio.ScalarImage(
        tensor=torch.zeros(reference_tensor_shape),
        affine=meta_info["original_subject_affine"]
    )

    resampler_to_original_grid = tio.Resample(
        target=reference_image_original_space,
        image_interpolation='nearest'
    )

    pred_resampled = resampler_to_original_grid(pred_label_map_roi_space)
    final_pred_numpy = pred_resampled.data.squeeze(0).cpu().numpy()

    return final_pred_numpy.transpose(2, 1, 0).astype(np.uint8)


def save_prediction(pred_arr, output_path, meta_info):
    """保存预测结果为NIfTI"""
    out_img = sitk.GetImageFromArray(pred_arr)
    original_sitk_image = meta_info.get("sitk_image_object")
    if original_sitk_image:
        out_img.SetOrigin(original_sitk_image.GetOrigin())
        out_img.SetDirection(original_sitk_image.GetDirection())
        out_img.SetSpacing(original_sitk_image.GetSpacing())
    sitk.WriteImage(out_img, output_path)
    print(f"保存预测结果: {output_path}")


def segment_spine_mri(model, img_path, output_path, point_coords=None, point_labels=None):
    """
    分割脊柱MRI

    Args:
        model: SAM-Med3D模型
        img_path: 输入图像路径
        output_path: 输出分割结果路径
        point_coords: 提示点坐标列表 [[x1,y1,z1], [x2,y2,z2], ...]
                      如果为None，使用图像中心点
        point_labels: 提示点标签列表 [1, 1, 0, ...]  (1=正样本, 0=负样本)
    """
    print(f"\n处理: {img_path}")

    # 预处理
    roi_image, meta_info = preprocess_image(img_path)
    print(f"  ROI形状: {roi_image.shape}")

    # 如果没有指定提示点，使用中心点
    if point_coords is None:
        center = [roi_image.shape[4]//2, roi_image.shape[3]//2, roi_image.shape[2]//2]
        point_coords = [center]
        point_labels = [1]
        print(f"  使用中心点: {center}")

    # 推理
    roi_pred = infer_with_point(model, roi_image, point_coords, point_labels)
    print(f"  预测体积中非零体素: {np.sum(roi_pred > 0)}")

    # 后处理
    final_pred = postprocess_mask(roi_pred, meta_info)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_prediction(final_pred, output_path, meta_info)

    return final_pred


def main():
    """主函数：处理所有病人的T2 sag序列"""
    # 加载模型
    model = load_model()

    # 输入输出目录
    nifti_dir = "/home/nyuair/junyi/SAM-Med3D/9hospital/nifti"
    output_dir = "/home/nyuair/junyi/SAM-Med3D/9hospital/segmentation_results"
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有病人目录
    patient_dirs = sorted(glob(os.path.join(nifti_dir, "*")))

    for patient_dir in patient_dirs:
        patient_name = os.path.basename(patient_dir)
        patient_output_dir = os.path.join(output_dir, patient_name)
        os.makedirs(patient_output_dir, exist_ok=True)

        # 找T2 sagittal序列 (最适合脊柱分割)
        t2_sag_files = glob(os.path.join(patient_dir, "*t2*sag*.nii.gz"))
        if not t2_sag_files:
            t2_sag_files = glob(os.path.join(patient_dir, "*sag*.nii.gz"))
        if not t2_sag_files:
            t2_sag_files = glob(os.path.join(patient_dir, "*.nii.gz"))

        print(f"\n{'='*60}")
        print(f"病人: {patient_name}")
        print(f"找到 {len(t2_sag_files)} 个序列文件")

        for img_path in t2_sag_files[:2]:  # 每个病人处理前2个序列
            seq_name = os.path.basename(img_path).replace('.nii.gz', '')
            output_path = os.path.join(patient_output_dir, f"{seq_name}_seg.nii.gz")

            try:
                segment_spine_mri(model, img_path, output_path)
            except Exception as e:
                print(f"  错误: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
