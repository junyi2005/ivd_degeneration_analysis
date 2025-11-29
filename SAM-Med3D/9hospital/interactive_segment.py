#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交互式SAM-Med3D分割脚本
支持指定多个提示点进行精确分割
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

sys.path.insert(0, '/home/nyuair/junyi/SAM-Med3D')
import medim

# 全局模型缓存
_model_cache = None

def get_model(ckpt_path=None):
    """获取或加载模型（带缓存）"""
    global _model_cache
    if _model_cache is None:
        if ckpt_path is None:
            ckpt_path = "/home/nyuair/junyi/SAM-Med3D/ckpt/sam_med3d_turbo.pth"
        print(f"加载模型: {ckpt_path}")
        _model_cache = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
    return _model_cache


def preprocess_image(img_path, crop_size=128, target_spacing=(1.5, 1.5, 1.5)):
    """预处理图像，返回ROI张量和元信息"""
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
    """
    使用多个提示点进行迭代推理

    Args:
        model: SAM-Med3D模型
        roi_image: 预处理后的ROI图像张量
        point_coords: 提示点坐标列表 [[x1,y1,z1], ...]
        point_labels: 提示点标签列表 [1, 0, 1, ...]
        num_iterations: 迭代次数（可以改善边界）
    """
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

    return medsam_seg_mask, medsam_seg_prob


def postprocess_mask(roi_pred_numpy, meta_info):
    """将ROI预测映射回原始图像空间"""
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
    """保存预测结果"""
    out_img = sitk.GetImageFromArray(pred_arr)
    sitk_img = meta_info.get("sitk_image_object")
    if sitk_img:
        out_img.SetOrigin(sitk_img.GetOrigin())
        out_img.SetDirection(sitk_img.GetDirection())
        out_img.SetSpacing(sitk_img.GetSpacing())
    sitk.WriteImage(out_img, output_path)


def segment_with_prompt(img_path, output_path, point_coords, point_labels, num_iterations=3):
    """
    使用指定提示点分割图像

    Args:
        img_path: 输入NIfTI图像路径
        output_path: 输出分割结果路径
        point_coords: 提示点坐标 [[x1,y1,z1], [x2,y2,z2], ...]
                      坐标范围 0-127（在128x128x128的ROI空间中）
        point_labels: 对应的标签 [1, 1, 0, ...]
                      1=前景（要分割的区域）, 0=背景（不要的区域）
        num_iterations: 迭代次数

    Returns:
        final_pred: 分割结果numpy数组
        prob_map: 概率图
    """
    model = get_model()

    print(f"\n处理: {img_path}")
    roi_image, meta_info = preprocess_image(img_path)
    print(f"  ROI形状: {roi_image.shape}")
    print(f"  提示点数量: {len(point_coords)}")

    roi_pred, prob_map = infer_with_points(model, roi_image, point_coords, point_labels, num_iterations)
    print(f"  ROI中分割体素: {np.sum(roi_pred > 0)}")

    final_pred = postprocess_mask(roi_pred, meta_info)
    print(f"  最终分割体素: {np.sum(final_pred > 0)}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    save_prediction(final_pred, output_path, meta_info)
    print(f"  保存至: {output_path}")

    return final_pred, prob_map


def get_spine_prompt_points(image_shape=(128, 128, 128)):
    """
    生成适合脊柱分割的多个提示点
    脊柱通常位于图像中央偏后
    """
    h, w, d = image_shape
    points = []
    labels = []

    # 中心脊柱区域 - 正样本点
    for z in [d//4, d//2, 3*d//4]:  # 在不同深度
        points.append([w//2, h//2, z])      # 中心
        points.append([w//2, h//2-10, z])   # 稍微偏前
        labels.extend([1, 1])

    # 边缘区域 - 负样本点（背景）
    points.append([10, 10, d//2])           # 左上角
    points.append([w-10, 10, d//2])         # 右上角
    points.append([10, h-10, d//2])         # 左下角
    points.append([w-10, h-10, d//2])       # 右下角
    labels.extend([0, 0, 0, 0])

    return points, labels


def segment_all_spine_mri():
    """分割所有病人的脊柱MRI"""
    nifti_dir = "/home/nyuair/junyi/SAM-Med3D/9hospital/nifti"
    output_dir = "/home/nyuair/junyi/SAM-Med3D/9hospital/segmentation_results_improved"

    # 获取脊柱提示点
    points, labels = get_spine_prompt_points()
    print(f"使用 {len(points)} 个提示点进行分割")
    print(f"  正样本点: {sum(labels)}")
    print(f"  负样本点: {len(labels) - sum(labels)}")

    # 处理所有T2 sag序列
    for patient_dir in sorted(glob(os.path.join(nifti_dir, "*"))):
        patient = os.path.basename(patient_dir)
        print(f"\n{'='*60}")
        print(f"病人: {patient}")

        # 找T2 sag序列
        t2_files = glob(os.path.join(patient_dir, "*t2*sag*.nii.gz"))
        if not t2_files:
            t2_files = glob(os.path.join(patient_dir, "*sag*.nii.gz"))[:2]

        for img_path in t2_files[:2]:
            seq_name = os.path.basename(img_path).replace('.nii.gz', '')
            output_path = os.path.join(output_dir, patient, f"{seq_name}_seg.nii.gz")

            try:
                segment_with_prompt(img_path, output_path, points, labels, num_iterations=3)
            except Exception as e:
                print(f"  错误: {e}")


if __name__ == "__main__":
    segment_all_spine_mri()
