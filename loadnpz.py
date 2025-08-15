import numpy as np
import torch
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines import LoadAnnotations
import mmcv

@PIPELINES.register_module()
class MotovisTransform:
    """
    精确复现原始ADE20KSemantic的图像处理逻辑：
    1. 裁剪：[:, 480:980, :] (1920×1080 -> 1920×500)
    2. Resize：双线性插值到 1124×1124
    """

    def __init__(self, 
                 crop_height_start=480, 
                 crop_height_end=980,
                 target_size=(1124, 1124)):
        self.crop_height_start = crop_height_start
        self.crop_height_end = crop_height_end
        self.target_size = target_size

    def __call__(self, results):
        """执行完整的Motovis变换"""

        # 步骤1: 裁剪图像 [:, 480:980, :]
        img = results['img']
        h, w = img.shape[:2]

        # 验证输入尺寸
        if h != 1080 or w != 1920:
            print(f"⚠️ 警告: 期望1920×1080，得到{w}×{h}")

        # 裁剪图像
        img_cropped = img[self.crop_height_start:self.crop_height_end, :, :]

        # 步骤2: Resize到1124×1124 (双线性插值)
        img_resized = mmcv.imresize(
            img_cropped, 
            self.target_size, 
            interpolation='bilinear',
            return_scale=False
        )

        results['img'] = img_resized
        results['img_shape'] = img_resized.shape
        results['ori_shape'] = img_resized.shape

        # 处理分割标注 (如果存在)
        for key in results.get('seg_fields', []):
            if key in results and results[key] is not None:
                seg = results[key]

                # 裁剪分割图 [:, 480:980]
                if seg.shape[0] == h:  # 如果是原始尺寸
                    seg_cropped = seg[self.crop_height_start:self.crop_height_end, :]
                else:
                    seg_cropped = seg  # 已经是裁剪后的尺寸

                # Resize分割图 (最近邻插值)
                seg_resized = mmcv.imresize(
                    seg_cropped,
                    self.target_size,
                    interpolation='nearest',
                    return_scale=False
                )

                results[key] = seg_resized

        return results



@PIPELINES.register_module()
class LoadMotovisNPZ:
    """
    加载NPZ文件并精确复现target_parser逻辑
    """

    def __init__(self, 
                 crop_height_start=480, 
                 crop_height_end=980,
                 reduce_zero_label=False):
        self.crop_height_start = crop_height_start
        self.crop_height_end = crop_height_end
        self.reduce_zero_label = reduce_zero_label

        # 16个通道的类别映射
        self.class_mapping = {i: i+1 for i in range(16)}

    def __call__(self, results):
        """加载NPZ并转换为分割标注"""
        # print(results.keys())
        if 'ann_info' in results:
            filename = results['ann_info']['seg_map']
        elif 'ann' in results['img_info']:
            filename = results['img_info']['ann']['seg_map']
        else:
            raise ValueError('Cannot find seg_map in results')
        
        full_path = results['seg_prefix'] +  '/' + filename

        # 加载NPZ文件
        data = np.load(full_path)
        annotations = data['arr_0']  # shape: (16, 1080, 1920)

        # 验证shape
        assert annotations.shape == (16, 1080, 1920), f"期望(16,1080,1920)，得到{annotations.shape}"

        # 步骤1: 裁剪 [:, 480:980, :] -> (16, 500, 1920)
        annotations_cropped = annotations[:, self.crop_height_start:self.crop_height_end, :]

        # 步骤2: 转换为分割图 (复现target_parser逻辑)
        segmap = self.npz_to_segmap(annotations_cropped)
        # segmap = np.ascontiguousarray(segmap.astype(np.uint8))

        # 存储结果
        results['gt_semantic_seg'] = segmap
        results['seg_fields'] = ['gt_semantic_seg']

        return results

    def npz_to_segmap(self, target):
        """
        精确复现原始target_parser逻辑：
        将16通道NPZ转换为单通道分割图
        """
        ch, h, w = target.shape  # (16, 500, 1920)
        segmap = np.zeros((h, w), dtype=np.uint8)

        # 初始化background mask
        background_mask = np.ones((h, w), dtype=bool)

        # 处理所有16个通道
        for channel_id in range(16):
            class_id = channel_id + 1  # 通道0->类别1, ..., 通道15->类别16

            # 获取当前通道的mask
            channel_mask = (target[channel_id] == 1)

            # 分配类别标签
            segmap[channel_mask] = class_id

            # 更新background mask
            background_mask = background_mask & (~channel_mask)

        # 设置background区域为0
        segmap[background_mask] = 0

        return segmap