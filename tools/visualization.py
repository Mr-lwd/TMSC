import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os

# def visualization(image_path, pred, mask, category, result_path):
#     cmap = cv2.COLORMAP_VIRIDIS
#     for j in range(pred.shape[0]):
#         pred_colored = np.uint8(np.clip(pred[j] * 255, 0, 255))
#         image = cv2.imread(image_path[j])
#         image = cv2.resize(image, (512, 512))
#         pred_colored = cv2.applyColorMap(pred_colored, cv2.COLORMAP_JET)
#         mask_colored = cv2.applyColorMap(mask[j].astype(np.uint8) * 255, cv2.COLORMAP_JET)

#         # 创建一个空的画布，宽度为三张图片的宽度之和，高度为最大高度
#         canvas_width = image.shape[1] * 3
#         canvas_height = image.shape[1]
#         canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
#         image_colored = image

#         # 将三张图片放入画布中
#         canvas[: image_colored.shape[0], : image_colored.shape[1], :] = image_colored
#         canvas[: image_colored.shape[0], image_colored.shape[1] : image_colored.shape[1]*2, :] = mask_colored
#         canvas[: image_colored.shape[0], image_colored.shape[1]*2 : image_colored.shape[1]*3, :] = pred_colored


#         # 从image_path提取文件名信息
#         filename = os.path.basename(image_path[j])  # 获取文件名
#         name_without_ext = os.path.splitext(filename)[0]  # 去掉扩展名
        
#         # 保存拼接后的图片
#         os.makedirs(f"{result_path}/{category}/{image_path[j].split('/')[-2]}", exist_ok=True)
#         cv2.imwrite(f"{result_path}/{category}/{image_path[j].split('/')[-2]}/{category}_{image_path[j].split('/')[-2]}_{name_without_ext}.png", canvas)

def visualization(image_path, pred, mask, category, result_path):
    for j in range(pred.shape[0]):
        img = cv2.imread(image_path[j])
        img = cv2.resize(img, (512, 512))

        # 预测热力图叠加
        pred_norm = np.clip(pred[j], 0, 1)
        pred_uint8 = np.uint8(pred_norm * 255)
        pred_color = cv2.applyColorMap(pred_uint8, cv2.COLORMAP_JET)
        pred_overlay = cv2.addWeighted(img, 0.5, pred_color, 0.5, 0)

        # GT 边界：根据 mask 二值图提取轮廓并绘制红色粗边
        m = (mask[j] > 0).astype(np.uint8)
        m_uint8 = (m * 255).astype(np.uint8)
        contours, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gt_overlay = img.copy()
        cv2.drawContours(gt_overlay, contours, -1, (0, 0, 255), thickness=2)

        # 画布：左为 GT 边界叠加，右为预测热力图叠加
        canvas = np.zeros((512, 1024, 3), dtype=np.uint8)
        canvas[:, :512, :] = gt_overlay
        canvas[:, 512:, :] = pred_overlay

        filename = os.path.basename(image_path[j])
        name_without_ext = os.path.splitext(filename)[0]
        out_dir = f"{result_path}/{category}/{image_path[j].split('/')[-2]}"
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(f"{out_dir}/{category}_{image_path[j].split('/')[-2]}_{name_without_ext}.png", canvas)